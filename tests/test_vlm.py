"""Multimodal routing, processor boundaries and cache isolation without Metal."""

import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from jev_mlx import BackendOutput, Candidate, DecisionRequest, MLXVLMDecisionEngine
from jev_mlx.backends.mlx_vlm import MLXVLMBackend, _PromptTokenizer
from jev_mlx.backends.vlm_loaders import load_local_vlm
from jev_mlx.prompt import CompiledChoice, PreparedPrompt


def request():
    return DecisionRequest({}, "Choose red", (Candidate("red", "Choose red", "red"),))


def test_images_stay_with_each_call_and_use_shared_result_validation(monkeypatch):
    calls = []

    class Backend:
        identity = {"name": "test"}

        def __init__(self, *args, **kwargs):
            pass

        def score(self, value, **kwargs):
            calls.append((value, kwargs))
            return BackendOutput({"red": 3.0, "__no_match__": 0.0, "__abstain__": -1.0})

    monkeypatch.setattr("jev_mlx.backends.mlx_vlm.MLXVLMBackend", Backend)
    engine = MLXVLMDecisionEngine("local", margin_threshold=0.1)
    value = request()
    result = engine.decide(value, images=["first.png"], cache_key="one")
    engine.decide(request(), images=["second.png"], use_cache=False)
    engine.decide(request())
    assert result.candidate_id == "red" and result.selected_value == "red"
    assert engine.model_identity == {"name": "test"}
    assert [c[1]["images"] for c in calls] == [["first.png"], ["second.png"], ()]
    assert calls[0][1]["cache_key"] == "one"
    assert calls[1][1]["use_cache"] is False
    assert calls[0][0] is not value and calls[0][0] == value
    calls.clear()
    with pytest.raises(TypeError, match="DecisionRequest"):
        engine.decide({"state": {}})
    assert not calls


def test_vlm_template_keeps_policy_and_user_data_with_image_count():
    tokenizer = SimpleNamespace(encode=Mock(return_value=[3]), decode=Mock(return_value="0"))
    processor = SimpleNamespace(tokenizer=tokenizer)
    apply = Mock(return_value="formatted")
    proxy = _PromptTokenizer(processor, {"model_type": "test"}, 2, apply)
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "data"}]
    assert proxy.apply_chat_template(messages) == "formatted"
    actual = apply.call_args.args[2]
    assert actual[0]["content"].startswith("policy")
    assert "untrusted" in actual[0]["content"]
    assert actual[1] == messages[1]
    assert messages[0]["content"] == "policy"
    assert apply.call_args.kwargs == {
        "num_images": 2, "add_generation_prompt": True, "enable_thinking": False,
    }
    assert proxy.encode("0") == [3] and proxy.decode([3]) == "0"


class Tokens:
    ndim = 2

    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, key):
        return SimpleNamespace(tolist=lambda: self.values[key[1]])


@pytest.fixture
def backend(monkeypatch):
    backend = MLXVLMBackend.__new__(MLXVLMBackend)
    backend.processor = object()
    backend.config = {"model_type": "fake"}
    backend.max_prompt_tokens = 64
    backend.identity = {"name": "fake"}
    backend.load_ms = 0
    backend._lock = threading.RLock()
    backend._images = Mock(return_value=[])
    backend._apply_template = Mock()
    backend._make_cache = Mock(side_effect=lambda model: [SimpleNamespace(state=[])])
    backend._prepare_inputs = Mock(return_value={
        "input_ids": Tokens(list(range(20))), "attention_mask": "mask",
        "pixel_values": "pixels", "image_grid_thw": "grid",
    })
    chosen = Mock()
    chosen.astype.return_value = chosen
    chosen.tolist.return_value = [3.0, 0.0, -1.0]
    output = MagicMock()
    output.logits.__getitem__.return_value.__getitem__.return_value = chosen
    backend.model = Mock(return_value=output)
    backend.mx = SimpleNamespace(
        array=lambda x: x, float32=object(), eval=Mock(), synchronize=Mock(),
        get_peak_memory=lambda: 1,
    )
    choices = tuple(
        CompiledChoice(id_, str(i), i, "choice")
        for i, id_ in enumerate(("red", "__no_match__", "__abstain__"))
    )
    monkeypatch.setattr(
        "jev_mlx.backends.mlx_vlm.prepare_prompt",
        lambda *args: PreparedPrompt(tuple(range(20)), (0,), (0, 1), choices),
    )
    return backend


def test_processor_fields_reach_model_and_every_call_uses_fresh_cache(backend):
    first = backend.score(request(), cache_key="same", use_cache=True)
    second = backend.score(request(), cache_key="same", use_cache=True)
    assert first.raw_scores == {"red": 3.0, "__no_match__": 0.0, "__abstain__": -1.0}
    assert first.cache == second.cache
    assert not first.cache["enabled"] and first.cache["reused_tokens"] == 0
    assert first.cache["prompt_tokens"] == 20
    a, b = backend.model.call_args_list
    assert a.kwargs["cache"] is not b.kwargs["cache"]
    assert a.kwargs["image_grid_thw"] == "grid"
    assert a.kwargs["pixel_values"] == "pixels" and a.kwargs["mask"] == "mask"
    assert backend._make_cache.call_count == 2


@pytest.mark.parametrize("values,error", [
    (list(range(65)), "Expanded"),
    ([*range(19), 99], "suffix changed"),
    ([], "nonempty"),
])
def test_processor_boundaries_fail_before_forward(backend, values, error):
    backend._prepare_inputs.return_value["input_ids"] = Tokens(values)
    with pytest.raises(ValueError, match=error):
        backend.score(request())
    backend.model.assert_not_called()


def test_image_input_is_bounded_local_and_copied():
    Image = pytest.importorskip("PIL.Image")
    backend = MLXVLMBackend.__new__(MLXVLMBackend)
    backend.max_image_pixels = 16
    original = Image.new("RGB", (4, 4), "red")
    copied = backend._images([original])[0]
    original.putpixel((0, 0), (0, 0, 255))
    assert copied.getpixel((0, 0)) == (255, 0, 0)
    for images, error in [
        ([original] * 5, "at most four"),
        ([Image.new("RGB", (5, 4))], "pixels"),
        (["https://example.com/image.png"], "local file"),
        ([object()], "local path"),
        ("image.png", "list or tuple"),
    ]:
        with pytest.raises(ValueError, match=error):
            backend._images(images)


def test_cli_routes_images_and_rejects_images_for_text_backend(monkeypatch, tmp_path, capsys):
    from jev_mlx.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["decide", "--model", "local", "--image", "red.png"])
    assert exc.value.code == 2
    fake = Mock()
    fake.decide.return_value.to_dict.return_value = {"status": "selected", "candidate_id": "red"}
    constructor = Mock(return_value=fake)
    monkeypatch.setattr("jev_mlx.vlm.MLXVLMDecisionEngine", constructor)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request().to_dict()))
    assert main([
        "decide", "--model", "local", "--backend", "mlx-vlm", "--request", str(path),
        "--image", "red.png", "--image", "blue.png",
    ]) == 0
    constructor.assert_called_once_with("local")
    assert fake.decide.call_args.kwargs["images"] == ["red.png", "blue.png"]
    assert json.loads(capsys.readouterr().out)["candidate_id"] == "red"


def test_generic_loader_is_local_and_disables_remote_code(monkeypatch, tmp_path):
    load = Mock(return_value=("model", "processor"))
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(load=load))
    assert load_local_vlm(tmp_path, {"model_type": "qwen3_vl"}) == ("model", "processor")
    load.assert_called_once_with(str(tmp_path), trust_remote_code=False, local_files_only=True)


def test_idefics_adapter_preserves_config_and_uses_official_components(monkeypatch, tmp_path):
    config_file = tmp_path / "processor_config.json"
    original = '{"processor_class":"Idefics3Processor","image_seq_len":64}'
    config_file.write_text(original)
    tokenizer = SimpleNamespace(chat_template="template")
    auto_tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    image_processor = SimpleNamespace(from_pretrained=Mock(return_value="pil"))
    processor = Mock(return_value="processor")
    model = Mock(return_value="weights")
    generic_load = Mock(side_effect=AssertionError("do not reload through AutoProcessor"))
    modules = {
        "mlx_vlm": SimpleNamespace(load=generic_load),
        "mlx_vlm.utils": SimpleNamespace(load_model=model),
        "transformers": SimpleNamespace(AutoTokenizer=auto_tokenizer),
        "transformers.models.idefics3.image_processing_pil_idefics3": SimpleNamespace(
            Idefics3ImageProcessorPil=image_processor,
        ),
        "transformers.models.idefics3.processing_idefics3": SimpleNamespace(
            Idefics3Processor=processor,
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    assert load_local_vlm(tmp_path, {"model_type": "idefics3"}) == ("weights", "processor")
    processor.assert_called_once_with(
        image_processor="pil", tokenizer=tokenizer, image_seq_len=64, chat_template="template",
    )
    model.assert_called_once_with(tmp_path, trust_remote_code=False, local_files_only=True)
    assert config_file.read_text() == original
