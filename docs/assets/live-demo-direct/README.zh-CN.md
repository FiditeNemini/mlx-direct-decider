# MLX Direct Decider 桌面演示录像

[返回项目首页](../../../README.zh-CN.md)

于 2026-10-14 使用界面中显示 **MLX Direct Decider** 的当前版本重新录制。
八条固定请求通过本地 Qwen3.5-9B 模型驱动真实 DOM 控件。
八次模型选择与实际执行检查均通过，没有浏览器或录制错误。
这是说明接入方式的演示，与留出的模型 benchmark 分开。

![MLX Direct Decider：完整本地模型演示，原速播放](preview.gif)

[完整 MP4](full-run.mp4) · [原始 WebM](original.webm) ·
[原始记录](transcript.json) · [材料哈希](provenance.json)

| 请求 | 模型原始选择 | 当次决策耗时 |
| --- | --- | ---: |
| Close it | `__no_match__` | 2445.6 ms |
| Close it | `close.notes` | 443.6 ms |
| First one | `play.orbit` | 2471.6 ms |
| Pause it | `player.pause` | 1525.4 ms |
| Did you just close it? | `__no_match__` | 2533.6 ms |
| Close it | `close.player` | 553.8 ms |
| Close it | `close.library` | 463.1 ms |
| Close it | `__no_match__` | 546.8 ms |

这些是本次录像各次请求的耗时，不是延迟分布。
模型在录制前已经加载，前缀缓存也已被前一轮浏览器采集使用，不能视为启动或冷缓存测量。
原始缓存记录、手动准备点击、模型响应和执行回执全部保留。
拒绝场景允许 `no_match` 或 `abstain`；本次模型均返回 `no_match`。

MP4 与 GIF 都以 **1 倍原速**保留完整时间线，包括推理等待和展示停顿。
只改变编码、GIF 尺寸、帧率和颜色数；模型输出与原始截图均未修图。
桌面与播放器是使用允许列表控件的虚构公开示例。

录制使用已提交代码的干净导出与本次名称修正补丁，未带入无关的工作区改动。
详见[源码身份](source.json)、[名称修正补丁](source.patch)及记录中的源码文件哈希。
更名前的录像完整保留在[历史报告](../live-demo/README.zh-CN.md)。

[复现录像](../../http-and-demo.zh-CN.md#interactive-showcase)。
