"""Canonical naming must preserve legacy object identity and CLI compatibility."""

import importlib
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

import jev_mlx
import mlx_direct_decider as sdk
import mlx_zeroeffort_decider as previous_sdk


@pytest.mark.parametrize("name", [
    "engine", "types", "session", "prompt", "vlm", "server", "demo", "token_codes",
    "backends.mlx_lm", "backends.mlx_vlm", "backends.vlm_loaders",
])
def test_submodules_share_the_same_objects(name):
    assert importlib.import_module(f"mlx_direct_decider.{name}") is importlib.import_module(
        f"jev_mlx.{name}"
    )
    assert importlib.import_module(f"mlx_zeroeffort_decider.{name}") is importlib.import_module(
        f"mlx_direct_decider.{name}"
    )


def test_requests_and_results_cross_namespaces():
    assert sdk.__all__ == jev_mlx.__all__
    for name in sdk.__all__:
        assert getattr(sdk, name) is getattr(jev_mlx, name)
        assert getattr(sdk, name) is getattr(previous_sdk, name)
    backend = Mock()
    backend.score.return_value = jev_mlx.BackendOutput(
        {"choose": 3.0, jev_mlx.NO_MATCH_ID: 0.0, jev_mlx.ABSTAIN_ID: -1.0}
    )
    request = jev_mlx.DecisionRequest({}, "Choose", (sdk.Candidate("choose", "Choose", True),))
    result = sdk.MLXDecisionEngine(backend=backend).decide(request)
    assert isinstance(result, jev_mlx.DecisionResult)
    assert result.candidate_id == "choose" and result.selected_value is True


@pytest.mark.parametrize("canonical,previous,legacy,explicit,expected", [
    ("new-local", "previous-local", "old-local", None, "new-local"),
    (None, "previous-local", "old-local", None, "previous-local"),
    (None, None, "old-local", None, "old-local"),
    ("new-local", "previous-local", "old-local", "explicit-local", "explicit-local"),
])
def test_model_environment_precedence(monkeypatch, tmp_path, capsys,
                                      canonical, previous, legacy, explicit, expected):
    from mlx_direct_decider.cli import main

    for name, value in (
        ("MLX_DIRECT_MODEL", canonical),
        ("MLX_ZEROEFFORT_MODEL", previous),
        ("JEV_MLX_MODEL", legacy),
    ):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    engine = Mock()
    engine.decide.return_value.to_dict.return_value = {"status": "selected"}
    constructor = Mock(return_value=engine)
    monkeypatch.setattr("jev_mlx.engine.MLXDecisionEngine", constructor)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(sdk.DecisionRequest({}, "Choose", ()).to_dict()))
    args = ["decide", "--request", str(path)]
    if explicit is not None:
        args += ["--model", explicit]
    assert main(args) == 0
    constructor.assert_called_once_with(expected)
    assert json.loads(capsys.readouterr().out)["status"] == "selected"


def test_imports_and_help_do_not_initialize_mlx():
    program = """
import sys
import mlx_direct_decider
from mlx_direct_decider.backends.mlx_lm import MLXLMBackend
from mlx_direct_decider.backends.mlx_vlm import MLXVLMBackend
assert not any(name in sys.modules for name in ('mlx.core', 'mlx_lm', 'mlx_vlm'))
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [sys.executable, "-m", "mlx_direct_decider", "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "mlx-direct-decider" in result.stdout
    assert "No fine-tuning required" in result.stdout
