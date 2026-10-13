"""Small synthetic runtime check, not a held-out vision quality benchmark."""

import argparse
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from PIL import Image

    from jev_mlx import Candidate, DecisionRequest, MLXVLMDecisionEngine

    engine = MLXVLMDecisionEngine(args.model)
    colors = (
        Candidate("red", "The image is predominantly red", "red"),
        Candidate("blue", "The image is predominantly blue", "blue"),
    )
    request = DecisionRequest(
        state={}, utterance="Classify the attached image using the allowed colors.",
        question="What is the dominant color in the attached image?", candidates=colors,
    )
    red, blue = (Image.new("RGB", (224, 224), c) for c in ("red", "blue"))
    boolean = DecisionRequest.boolean(
        state={}, utterance="Answer the question using the attached image.",
        question="Is the image predominantly red?",
    )
    cases = [
        ("text_only", DecisionRequest(
            {}, "Choose red", (Candidate("red", "Choose red"), Candidate("blue", "Choose blue")),
        ), [], "red"),
        ("red", request, [red], "red"),
        ("blue_changed_image", replace(request, state_version=1), [blue], "blue"),
        ("red_repeated", request, [red], "red"),
        ("red_reversed_choices", replace(request, candidates=colors[::-1]), [red], "red"),
        ("boolean_red", boolean, [red], "true"),
        ("boolean_blue", boolean, [blue], "false"),
    ]
    records = []
    for name, value, images, expected in cases:
        result = engine.decide(value, images=images)
        record = {
            "case": name, "request": value.to_dict(), "expected": expected,
            "correct": result.candidate_id == expected, "result": result.to_dict(),
        }
        records.append(record)
        print(json.dumps({
            "case": name, "expected": expected, "actual": result.raw_selected_id,
            "correct": record["correct"], "ms": result.timing["total_ms"],
        }), flush=True)
    repeated = [records[i]["result"]["raw_scores"] for i in (1, 3)]
    parity_delta = max(abs(repeated[0][k] - repeated[1][k]) for k in repeated[0])
    sources = [
        Path("src/jev_mlx/backends/mlx_vlm.py"), Path("src/jev_mlx/vlm.py"),
        Path("src/jev_mlx/prompt.py"), Path("src/jev_mlx/engine.py"), Path(__file__),
        Path("src/jev_mlx/token_codes.py"), Path("src/jev_mlx/backends/vlm_loaders.py"),
    ]
    report = {
        "scope": "Seven synthetic development smoke cases; not general vision accuracy",
        "hardware": {"machine": platform.machine(), "system": platform.platform()},
        "model": engine.model_identity,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "weights_rehashed": False,
        "correct": sum(r["correct"] for r in records), "cases": len(records),
        "repeated_red_max_logit_difference": parity_delta,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    assert all(not r["result"]["cache"]["hit"] for r in records)
    assert all(r["result"]["cache"]["reused_tokens"] == 0 for r in records)
    assert parity_delta <= 0.01, parity_delta


if __name__ == "__main__":
    main()
