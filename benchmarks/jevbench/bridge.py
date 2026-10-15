"""JevBench adapter extracted from the measured serial local experiment.

Portability-only edits remove private model/setup paths and clarify provenance.
Candidate mapping, softmax, native rejection and scoring are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

from mlx_direct_decider import Candidate, DecisionRequest, MLXDecisionEngine

ROOT = Path(__file__).resolve().parent
MODEL = "mlx-community/Qwen3.5-9B-OptiQ-4bit"
POLICY = {
    "max_prompt_tokens": 4096,
    "max_candidates": 64,
    "prefill_step_size": 256,
    "cache_bytes": 512 * 1024**2,
    "cache_entries": 16,
    "margin_threshold": 0.0,
    "use_cache": True,
    "mapping": "One native decision per question, input state wrapped unchanged under input; "
    "question placement is controlled by the explicit benchmark profile; candidates retain "
    "input order and exact descriptions. Score levels are enum candidates. "
    "No truncation, filtering or retries. Prompt development is disclosed in docs/jevbench.md.",
    "probabilities": "Uncalibrated native logits, softmax conditional on supplied business "
    "options, only when native engine selects an option. Native no_match/abstain preserved "
    "as unanswered rather than forced to a business choice.",
}


class CapacityError(ValueError):
    pass


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def make_request(state, question, labels=None):
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        labels = ["no", "yes"]
        criteria = criteria if isinstance(criteria, dict) else {}
        descriptions = [criteria.get("false", "The proposition is false."),
                        criteria.get("true", "The proposition is true.")]
    elif kind == "choice":
        labels = list(labels) if labels is not None else list(criteria)
        descriptions = [(criteria or {}).get(label, label) for label in labels]
    elif kind == "score":
        labels = list(labels) if labels is not None else [str(i) for i in range(len(criteria))]
        descriptions = [criteria[int(label)] for label in labels]
    else:
        raise CapacityError(f"Unsupported typed question: {kind}")
    if not 1 <= len(labels) <= POLICY["max_candidates"]:
        raise CapacityError(f"options per choice {len(labels)} exceeds limit 64")
    try:
        candidates = tuple(Candidate(f"option_{i:03d}", f"{label}: {text(desc)}",
                                     label == "yes" if kind == "noul" else label)
                           for i, (label, desc) in enumerate(zip(labels, descriptions)))
        request = DecisionRequest(
            state={"input": state},
            utterance=text(question["instructions"]),
            question="Answer the question using the supplied input and allowed answers.",
            candidates=candidates,
            kind="boolean" if kind == "noul" else "enum",
        )
    except ValueError as exc:
        raise CapacityError(f"Declared native request size/type limit: {exc}") from exc
    return request, dict(zip((c.id for c in candidates), labels))


def conditional_probabilities(result, mapping):
    logits = {label: result.raw_scores[cid] for cid, label in mapping.items()}
    top = max(logits.values())
    weights = {label: math.exp(score - top) for label, score in logits.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


class LocalModel:
    def __init__(self, model=MODEL):
        self.engine = MLXDecisionEngine(
            model, max_prompt_tokens=4096, prefill_step_size=256,
            cache_bytes=512 * 1024**2, cache_entries=16, margin_threshold=0.0,
        )
        self.identity = self.engine.model_identity
        self.model_name = self.identity["name"]

    def decide(self, state, question, labels=None):
        request, mapping = make_request(state, question, labels)
        try:
            result = self.engine.decide(request, use_cache=True, cache_key="external-benchmarks")
        except ValueError as exc:
            if "Prompt exceeds" in str(exc) or "candidate description" in str(exc):
                raise CapacityError(str(exc)) from exc
            raise
        return result, mapping, request

    def metadata(self):
        return {"identity": self.identity, "policy": POLICY,
                "hardware": {"machine": platform.machine(), "platform": platform.platform()},
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class JevAdapter:
    name = "mlx-direct-decider-qwen35-9b-optiq4"
    price_input_per_m = None
    price_output_per_m = None
    cost_basis = "local_apple_silicon_unmetered_compute_not_free"

    def __init__(self, local):
        self.local = local

    def reserve_estimate(self, task):
        return 0.0

    def run(self, task):
        from jevbench.adapters.base import DecisionResult

        started = time.perf_counter()
        try:
            result, mapping, request = self.local.decide(task.state, task.question, task.labels)
            raw = result.to_dict()
            raw["runtime"] = {"policy": POLICY, "device": "apple_silicon_metal",
                              "profile": getattr(self.local, "profile", "released")}
            common = dict(adapter=self.name, model=self.local.model_name,
                          latency_s=time.perf_counter() - started, raw=raw,
                          request_body=request.to_dict(),
                          usage={"input_tokens": result.cache.get("prompt_tokens"),
                                 "output_tokens": 0})
            if result.status != "selected":
                return DecisionResult(ok=True, probs=None,
                    probs_source="label_only_no_calibrated_distribution", label=None,
                    error=f"native {result.status}", **common)
            return DecisionResult(ok=True, probs=conditional_probabilities(result, mapping),
                                  probs_source="native", **common)
        except CapacityError as exc:
            return DecisionResult(adapter=self.name, ok=False, status=422,
                                  model=self.local.model_name, error=str(exc),
                                  latency_s=time.perf_counter() - started)
