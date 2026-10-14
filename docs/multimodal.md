# MLX, MLX-LM and MLX-VLM

[Back to home](../README.md)

MLX Direct Decider reuses compatible pretrained checkpoints on Apple Silicon, with no
fine-tuning, new decision head, or decision-specific weights. The model still
processes the full input. MLX Direct Decider reads verified candidate-token logits at the
final input position; it does not generate or parse an answer string.

| Component | Integration |
| :--- | :--- |
| [MLX](https://github.com/ml-explore/mlx) | Array/compute foundation used by both backends. |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | `MLXDecisionEngine`: text decisions with bounded prefix reuse. |
| [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | `MLXVLMDecisionEngine`: text or text + local images, with fresh context on every call. |

Metal is MLX's GPU execution layer on these Macs; it does not need another JEV
backend. Swift, Ollama/GGUF, audio/video inputs, embeddings, and app-specific
bridges are outside this integration. Installing the library does not attach it
to every application. Pass the local checkpoint you want to use.

## Install and run images

Install from this checkout on Apple Silicon:

```sh
python -m pip install -e '.[mlx,vlm]'
```

The tested runtime pins MLX 0.31.2, MLX-LM 0.31.3, MLX-VLM 0.6.4,
Transformers 5.9.0 and Tokenizers 0.22.2. MLX-VLM 0.7.8 requires a newer
Transformers release; it is not claimed compatible with this pinned environment.
Weights are loaded locally, with remote code disabled. Obtain an ordinary
compatible checkpoint separately if you do not already have one.

```python
from mlx_direct_decider import Candidate, DecisionRequest, MLXVLMDecisionEngine

engine = MLXVLMDecisionEngine("/absolute/path/to/Qwen3-VL-4B-Instruct-4bit")
request = DecisionRequest(
    state={},
    question="What is the dominant color in the attached image?",
    utterance="Classify the attached image using the allowed colors.",
    candidates=(
        Candidate("red", "The image is predominantly red", "red"),
        Candidate("blue", "The image is predominantly blue", "blue"),
    ),
)
result = engine.decide(request, images=["examples/synthetic-red.png"])
print(result.status, result.candidate_id, result.selected_value)
```

The CLI accepts the same JSON request and explicit local image paths:

```sh
mlx-direct-decider decide --backend mlx-vlm \
  --model /absolute/path/to/Qwen3-VL-4B-Instruct-4bit \
  --request examples/vision.json --image examples/synthetic-red.png
```

Omit `images` / `--image` for a text-only request through MLX-VLM. Reuse the loaded
engine for later requests. Boolean requests use `DecisionRequest.boolean(...)`
and return typed `True` / `False` values through the same result validation.
The existing `mlx-direct-decider serve` and browser demos continue to use MLX-LM and
structured text state; they do not accept image uploads.

## Bounds and cache behavior

- Images are per-call local paths or PIL images, copied to RGB before inference.
  URLs are not accepted. Defaults allow at most four images, each at most
  4,194,304 pixels; `max_image_pixels` is configurable.
- The 4,096-token default applies after visual placeholders expand too.
  Models/processors can impose further limits. The real-model smoke checks use
  one image at a time; four-image input is an API bound, not verified quality.
- Every VLM request uses a fresh official model cache and processes the full
  prompt. `cache.hit` is false and `reused_tokens` is zero, even with
  `use_cache=True`. Image pixel hashes identify the inputs; no answers or visual
  features are reused. Do not apply MLX-LM warm-prefix latency claims to images.
- Candidate limits, reserved rejection choices, finite-score checks and margin
  semantics are shared with the text engine. Scores are uncalibrated rankings;
  even simple images can produce wrong decisions.
- If an image represents application state, advance `state_version` when that
  state changes and revalidate before executing. The library does not observe
  screenshots or external state changes automatically.

## Real-model smoke results · 2026-10-13

Apple M2 Max, 64 GiB unified memory, ARM Python 3.13.2. Seven synthetic development
cases exercise text, solid red/blue images, changed and repeated images, reversed
candidate order, and boolean answers. These are runtime checks, not a frozen
vision benchmark or evidence of general screenshot understanding.

| Checkpoint | Runtime cases completed | Correct / 7 | Result |
| :--- | ---: | ---: | :--- |
| Qwen3-VL-4B-Instruct-4bit (`qwen3_vl`) | 7/7 | 7/7 | Preferred starting point for the image example. |
| SmolVLM-500M-Instruct-4bit (`idefics3`) | 7/7 | 3/7 | Loads and scores, but mistakes and order sensitivity make it unsuitable as the example default. |
| gemma-4-e4b-it-4bit (`gemma4`) | 0/7 | — | Blocked during upstream loading: audio convolution weight shape mismatch. Its recorded MLX-LM text results remain separate. |

Both completed runs retained identical logits for the repeated red image and
reported zero reused tokens for every request. The
[evidence manifest](../benchmarks/results/runtime-2026-10-13/manifest.json)
links full results, failures, revisions, configuration/input hashes and source
hashes. Complete weight-file hashes were not recomputed. Reproduce a VLM run:

```sh
python -m benchmarks.vlm_smoke --model /absolute/path/to/model \
  --output runs/my-vlm-smoke.json
```

## Compatibility changes

The VLM backend and Idefics3 adapter live in dedicated modules. For Idefics3,
Transformers 5.9's automatic image-processor path requires PyTorch/Torchvision;
the adapter constructs its official PIL image processor and processor directly,
preserving the checkpoint's image sequence length, template and weights.

One shared prompt change is necessary for Mistral v0.3: a letter's standalone
token ID can differ from its ID immediately after the assistant delimiter. The
dedicated token helper verifies the actual one-token continuation and round trip
and uses that ID. Codes, business IDs, policy, model forward, and cache boundaries
stay the same. Already valid token IDs stay the same; the added decode check
does not schedule model inference. This cannot be fixed in the model loader,
because the required token depends on the completed prompt.

The public connections are a lazy Python export and the CLI's explicit
`--backend mlx-vlm` / `--image` options. No changes to the shared decision engine,
request schema, execution rules, MLX-LM inference loop, or cache policy are needed.
