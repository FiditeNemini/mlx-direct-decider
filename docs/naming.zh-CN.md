# 项目名称与早期命名检索

[返回项目首页](../README.zh-CN.md)

项目名称为 **MLX ZeroEffort Decider**，仓库为
[ItsOdeLeo/MLX-ZeroEffort-Decider](https://github.com/ItsOdeLeo/MLX-ZeroEffort-Decider)。
作者于 **2026-10-13** 选定此名称。副标题明确写为：
**Decisions from existing models. No fine-tuning required.**
即：**让已有模型直接做决策，无需微调。**

| 对外标识 | 名称 |
| :--- | :--- |
| 发行包与 CLI | `mlx-zeroeffort-decider` |
| Python 导入 | `mlx_zeroeffort_decider` |
| 本地模型环境变量 | `MLX_ZEROEFFORT_MODEL` |
| 显式模型测试环境变量 | `MLX_ZEROEFFORT_TEST_MODEL` |

原有 `jev_mlx` 导入、`jev-mlx` 命令、`JEV_MLX_MODEL` 和 `JEV_TEST_MODEL` 继续兼容。
新旧 Python 命名空间共用相同类与实现，请求和会话可以交叉使用。
模型权重、推理行为与缓存策略没有变化。实现源码保留在 `src/jev_mlx`，
以维持现有源码链接和调用方兼容。更名不表示向 PyPI 发布包。

ZeroEffort 在这里指无需额外的 fine-tuning 流程；应用仍需提供兼容的本地检查点、状态与候选项。

## 实现与兼容性

新命名空间转接到现有实现，使请求／结果类在新旧导入之间保持同一对象身份。
包元数据与共用 CLI 注册新命令和模型环境变量；旧命令和变量继续可用，明确传入的
`--model` 优先。这些连接属于打包／参数解析层，不能放进某个模型适配模块。
可选后端的安装错误信息改用新发行包名，不影响评分。

服务端健康信息与当前界面显示新项目名；对健康信息 `project` 字符串做严格比较的调用方
应使用新名称。Benchmark 环境元数据记录新发行包版本。这些只改变身份信息／诊断文本，
不改变推理、调度或缓存。历史 benchmark 文件保持原样。命名空间身份、环境变量优先级、
不依赖可选运行库的导入及新旧 CLI 入口单独验证，与已记录的模型质量评测区分。

## 旧称与历史证据

旧称包括 **JEV MLX**、**JEVKit MLX** 与 **MLXJ**。2026-09-20 曾将展示名称从
MLXJ 改为 JEV MLX。录像、冻结协议及 benchmark 证据保留原始名称、命令、源码路径与文件字节；
旧复现命令仍可通过兼容入口运行。GitHub 仓库在新名称下保留原有提交历史与 issues。

项目仍是受 Jev 启发的独立实现，不是 TypeSafe AI 官方项目，也不是已有的
[bnsd55/jevmlx](https://github.com/bnsd55/jevmlx)。不使用 Jev 权重，也不复现未公开训练方法。

## 早期检索记录

检索日期：2026-09-19。下表保留早期公开搜索的时间快照，不代表当前命名决定、名称已被保留或注册平台保证接受该名称。

| 候选 | 本次核对结果 | 判断 |
| --- | --- | --- |
| JEV MLX / jevmlx | 已有同领域 [bnsd55/jevmlx](https://github.com/bnsd55/jevmlx) | 已有近似名称；请用作者与完整仓库地址区分 |
| JF | [PyPI 的 jf](https://pypi.org/project/jf/) 已是 JSONL 查询工具 | 太短，也丢失了 MLX 的识别度 |
| JF MLX | GitHub 仓库名搜索 `jf-mlx` / `jfmlx` 未检出结果 | 无法确认用户记忆中的具体项目，不能说绝无重名 |
| MLXJ | GitHub `mlxj in:name` 返回部分匹配，没有发现 basename 精确为 `mlxj` 的仓库；[MLXJ 账号](https://github.com/MLXJ) 已存在 | 早期展示名称，现已改为 JEV MLX，明确体现启发来源 |

当时 [PyPI mlxj JSON API](https://pypi.org/pypi/mlxj/json) 返回 404；`jf-mlx` 与 `jev-mlx` 的对应 API 也返回 404。这里只说明没有公开项目记录，不能保证名称可注册，也没有创建账号、仓库或发布包。

可以通过 [GitHub 官方仓库搜索 API](https://api.github.com/search/repositories?q=mlxj+in%3Aname)复查仓库名，搜索结果可能变化。

## README 参考

页面结构参考了这些项目的一手 README，未复制其图形、品牌或正文：

- [LobeHub](https://github.com/lobehub/lobehub#readme)：居中视觉标题、语言切换、社区与支持入口。
- [RAGFlow](https://github.com/infiniflow/ragflow#readme)：分语言 README、文档导航与清晰章节。
- [Clash Verge Rev](https://github.com/clash-verge-rev/clash-verge-rev#readme)：标识、语言入口和赞助信息。

语言只在主页选择：`README.md` 是默认英文首页，`README.zh-CN.md` 是中文首页。分别进入完整的同语言文档路径，内页不再提供语言切换。相对链接在 GitHub 上无需 JavaScript 即可使用。所有新视觉素材为本项目原创 SVG。

公开社交链接使用作者提供的账号信息。2026-10-08，作者将显示名称更新为**欧的Leo**，小红书号和 X 账号均更新为 **ItsOdeLeo**。README 和关注徽章使用 [X 上的 @ItsOdeLeo](https://x.com/ItsOdeLeo)，以及原有的[小红书主页链接](https://xhslink.com/m/18bjTTf180W)。旧名片印有过时账号信息，因此不再展示在首页；[社交资料来源](assets/social/README.zh-CN.md)保留了历史记录。赞助入口仍未配置。
