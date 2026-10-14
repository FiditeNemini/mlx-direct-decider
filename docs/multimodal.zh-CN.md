# MLX、MLX-LM 与 MLX-VLM

[返回项目首页](../README.zh-CN.md)

MLX Direct Decider 复用 Apple Silicon 上兼容的预训练检查点，无需 fine-tuning、新决策头或专用决策权重。模型仍需处理完整输入；MLX Direct Decider 读取最后输入位置上经过验证的候选 token logits，不生成或解析回答字符串。

| 组件 | 接入方式 |
| :--- | :--- |
| [MLX](https://github.com/ml-explore/mlx) | 两个后端共同使用的数组与计算底座。 |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | `MLXDecisionEngine`：文本决策，支持有容量上限的前缀复用。 |
| [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | `MLXVLMDecisionEngine`：文本或文本＋本地图片，每次重新计算上下文。 |

Metal 由 MLX 负责这些 Mac 上的 GPU 执行，不需要另写 JEV 后端。Swift、Ollama/GGUF、音视频输入、嵌入与具体应用桥接不在这次接入范围内。安装库不会自动接管所有应用，需要传入想使用的本地检查点。

## 安装和图片决策

在 Apple Silicon 上，从仓库安装：

```sh
python -m pip install -e '.[mlx,vlm]'
```

已测试环境固定为 MLX 0.31.2、MLX-LM 0.31.3、MLX-VLM 0.6.4、Transformers 5.9.0 与 Tokenizers 0.22.2。MLX-VLM 0.7.8 要求更新的 Transformers，这里不宣称它与当前固定环境兼容。本地加载权重并禁用远程代码；若没有现成模型，需先单独获取普通兼容检查点。

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

CLI 使用相同 JSON 请求，并明确传入本地图片路径：

```sh
mlx-direct-decider decide --backend mlx-vlm \
  --model /absolute/path/to/Qwen3-VL-4B-Instruct-4bit \
  --request examples/vision.json --image examples/synthetic-red.png
```

省略 `images` / `--image` 时，通过 MLX-VLM 做纯文本决策。后续请求复用已加载引擎。布尔请求使用 `DecisionRequest.boolean(...)`，通过相同结果校验返回有类型的 `True` / `False`。现有 `mlx-direct-decider serve` 与浏览器演示继续使用 MLX-LM 和结构化文本状态，不接受图片上传。

## 输入上限与缓存行为

- 图片按次传入本地路径或 PIL 图片，推理前复制成 RGB，不接受 URL。默认最多四张，每张最多 4,194,304 像素；可配置 `max_image_pixels`。
- 默认 4,096 token 上限也检查视觉占位符展开后的长度。模型与处理器还可能有其他限制。真实模型检查每次只使用一张图片；四张是 API 上限，不是已验证的质量范围。
- 每次 VLM 请求都创建全新的官方模型缓存，并处理完整提示词。即使 `use_cache=True`，`cache.hit` 也是 false、`reused_tokens` 也是零。记录图片像素哈希，不复用答案或视觉特征；不能把 MLX-LM 的热前缀延迟套用到图片。
- 候选上限、保留拒绝项、有限分数检查及 margin 语义与文本引擎共用。分数是未经校准的排名；简单图片也可能判断错误。
- 图片表示应用状态时，状态变化应更新 `state_version`，执行前再次校验。库不会自动观察截图或外部状态变化。

## 真实模型冒烟结果 · 2026-10-13

Apple M2 Max、64 GiB 统一内存、ARM Python 3.13.2。七个合成开发用例覆盖文本、纯红／纯蓝图片、换图与重复图片、候选顺序交换及布尔回答。这是运行检查，不是冻结的视觉 benchmark，也不能证明通用截图理解能力。

| 检查点 | 完成运行的用例 | 正确 / 7 | 结论 |
| :--- | ---: | ---: | :--- |
| Qwen3-VL-4B-Instruct-4bit（`qwen3_vl`） | 7/7 | 7/7 | 图片示例的优先起点。 |
| SmolVLM-500M-Instruct-4bit（`idefics3`） | 7/7 | 3/7 | 能加载与评分，但有错误及顺序敏感，不适合作为默认示例。 |
| gemma-4-e4b-it-4bit（`gemma4`） | 0/7 | — | 上游加载时出现音频卷积权重形状不匹配；已记录的 MLX-LM 文本结果单独保留。 |

两次完整运行中，重复红图得到相同 logits，每个请求均报告零复用 token。[证据清单](../benchmarks/results/runtime-2026-10-13/manifest.json)链接完整结果、失败、模型修订、配置／输入哈希及源码哈希；没有重新计算完整权重文件哈希。复现 VLM 检查：

```sh
python -m benchmarks.vlm_smoke --model /absolute/path/to/model \
  --output runs/my-vlm-smoke.json
```

## 兼容改动说明

VLM 后端与 Idefics3 适配放在专属模块。Transformers 5.9 对 Idefics3 的自动图片处理器路径要求 PyTorch/Torchvision，因此适配直接构造官方 PIL 图片处理器与 processor，保留检查点的图片序列长度、模板和权重。

一处公共提示词改动用于 Mistral v0.3：字母单独编码时的 token ID 可能与助手分隔符后的 ID 不同。专属 token 辅助模块验证实际的单 token 续写与往返一致性，并读取对应 ID。选项编码、业务 ID、策略、模型前向与缓存边界均保持原样。原本有效的 token ID 不变；新增解码检查不会触发模型推理。该问题不能只在模型加载器解决，因为所需 token 取决于完整提示词。

公共连接仅增加延迟导入的 Python 导出以及 CLI 的 `--backend mlx-vlm` / `--image` 选项。无需更改公共决策引擎、请求 schema、执行规则、MLX-LM 推理循环或缓存策略。
