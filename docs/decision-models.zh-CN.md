# JEV MLX 与 AnyJev：无需微调，让已有 LLM 直接做决策

[返回项目首页](../README.zh-CN.md)

这份说明帮助评估本地动作选择、分类和语义路由的开发者，将 JEV MLX 与
AnyJev、OpenDecider、Laya、Laya-MLX 放在同一技术领域中理解。它们的共同需求是：
把应用状态和问题转化为宿主代码能够使用的有限决策。

JEV MLX 是独立维护、面向已有因果 MLX-LM 模型的库。可选的 [MLX-VLM 接入](multimodal.zh-CN.md)使用兼容预训练视觉语言检查点提供文本／图片候选选择，同样无需微调。下列相关项目各自拥有独立
实现；链接说明技术相似点与备选方案，不代表厂商关联或检查点兼容性。

## 复用已有权重，无需额外训练

**JEV MLX 在兼容 MLX-LM 检查点上增加决策评分，无需训练或微调模型、增加训练后的决策头
或权重适配器，也不要求单独的决策模型权重。** 如果本地已有兼容基础权重，可以直接
复用；否则先获取基础检查点。后端加载本地目录，不负责下载权重。

完整模型处理提示词，并在可用时复用有效前缀缓存。保持不变的原始输出头给出
最后一个输入位置的下一 token logits。JEV MLX 取出经过分词器验证的选项编码
对应的 logits，再将结果映射回业务 ID。这省去了生成后续文本的步骤，不意味着
只运行最后一层。详见[模型推理路径](architecture.zh-CN.md)与
[公开评分实现](../src/jev_mlx/backends/mlx_lm.py)。

[OpenDecider](https://github.com/manjunathshiva/opendecider/blob/b830ea46961105da8c30e30a5f11ccce211bbf43/README.md)
和 [Laya](https://github.com/NandhaKishorM/laya/blob/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/README.md)
发布训练好的决策模型检查点，Laya-MLX 保留 Laya 的预训练权重。它们的用户可以
直接加载已发布权重，不必亲自训练。主要区别是**复用已有兼容基础模型权重，
还是加载专用决策模型检查点**，不应表述为其他项目的用户必须自行训练模型。

## 与 AnyJev 共同复用已有 LLM 的方向

[AnyJev](https://github.com/nokia-applied-research/AnyJev/blob/b03158e9ba18e84c01054aaa393481925088db07/README.md)
也提供无需训练的 `Decider` 路径。其 L0 方法包含候选轮换和标签先验校正；
独立的 `Tacit` 路径则使用自蒸馏检查点。二者共享从已有 LLM 读取决策的基本思路。

JEV MLX 聚焦 Apple Silicon 上的原生 MLX-LM 推理、动态业务 ID、明确拒绝、
有效前缀复用和版本化应用执行。直接评分未包含 AnyJev 的 L0 校正，候选分数
也不是经过校准的正确概率。查看的 AnyJev 后端目录包含 Hugging Face、vLLM、
SGLang 和 llama.cpp 实现，路线图将 MLX 列为后续后端工作。项目各自独立，
当前没有随本库提供相互适配器。

这是技术对照，不构成速度或准确率优势证明。本仓库尚无二者共同运行的基准测试。

## 实现方式对照

| 项目 | 模型与运行时方向 | 集成区别 |
| :--- | :--- | :--- |
| **JEV MLX** | 在 Apple Silicon 上复用已有兼容的因果 MLX-LM 权重，读取原始选项 token logits，无需额外训练。 | 动态候选项、稳定业务 ID、`no_match` / `abstain`、状态版本和单次执行授权。 |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev/blob/b03158e9ba18e84c01054aaa393481925088db07/README.md) | 无需训练地读取 LLM 输出，配合候选轮换和标签先验校正；另有自蒸馏模型。 | 自己的带类型问题约定及运行时后端；本库尚未提供 AnyJev 集成。 |
| [OpenDecider](https://github.com/manjunathshiva/opendecider/blob/b830ea46961105da8c30e30a5f11ccce211bbf43/README.md) | 开放权重的 System 1 决策模型系列，包含编码器与解码器路线。 | 自己的带类型问题约定、服务客户端及模型对应的运行时。 |
| [Laya](https://github.com/NandhaKishorM/laya/blob/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/README.md) · Convai Innovations | 基于编码器的决策模型，提供选择、有序评分和是非输出。 | 自己的问题结构、决策头与检查点路由。 |
| [Laya-MLX](https://github.com/mizorewww/laya-mlx/blob/ca5940aa9286dbbdfaacbecbdb7b337295ad36a2/README.md) | Laya 编码器和决策头的独立 MLX 实现。 | 面向 Laya 检查点的 Apple Silicon 本地推理，API 与 JEV MLX 分开。 |

相关项目的描述来自下方列出的上游 README。JEV MLX 的行为见
[架构](architecture.zh-CN.md)和 [Python API](python-api.zh-CN.md)。

## 如何围绕自己的应用评估

如果已有经过评估的 MLX-LM 检查点，需要从有限集合中选择应用允许的动作，
JEV MLX 提供相应的请求与执行约定。如果需要专门的决策模型系列，可以查看
OpenDecider 或 Laya 的模型卡和 API；如果希望用 MLX 在 Apple Silicon 上运行
Laya 模型，则查看 Laya-MLX 的转换与推理说明。这些是评估起点，不是模型质量排名。

用有代表性的应用状态和候选项，比较语义正确率、拒绝行为、延迟、内存与接入工作。
启动、权重已加载、前缀可复用、状态更新等条件应分开测量。硬件、模型、输入或
数据集不同的数字不能直接证明速度或准确率优势。JEV MLX 自己的
[评测方法](evaluation.zh-CN.md)与 [36 条扩展结果](extended-results.zh-CN.md)
提供了本项目的证据及局限。

## JEV MLX 能直接加载 OpenDecider 或 Laya 权重吗？

共同用途不等于加载兼容性。JEV MLX 当前提供 MLX-LM 后端，只评估了
[支持模型](models.zh-CN.md)中列出的本地检查点，尚未提供 OpenDecider 或 Laya
适配器及检查点验证。即使基础架构名称相似，也不足以证明分词器、提示词、
输出头或语义行为兼容。

## 这些分数的含义相同吗？

不同。JEV MLX 对候选 logits 计算受限 softmax 和原始 logit margin，
其分数**不是经过校准的正确概率**。OpenDecider 与 Laya 的上游文档描述了
经过校准的决策分布；这是上游的说明，不是 JEV MLX 的实测结论。未经应用自身
数据验证，不应在这些系统之间直接搬用分数或阈值。

## 来源快照

AnyJev 于 **2026-10-13** 对照
[README 修订 `b03158e`](https://github.com/nokia-applied-research/AnyJev/blob/b03158e9ba18e84c01054aaa393481925088db07/README.md)
及其[后端目录](https://github.com/nokia-applied-research/AnyJev/tree/b03158e9ba18e84c01054aaa393481925088db07/anyjev/backends)查看。

其他项目于 **2026-10-08** 查看的一手 README 修订如下：

- [OpenDecider · `b830ea4`](https://github.com/manjunathshiva/opendecider/blob/b830ea46961105da8c30e30a5f11ccce211bbf43/README.md)。
- [Laya · `3cf26cb`](https://github.com/NandhaKishorM/laya/blob/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/README.md)。
- [Laya-MLX · `ca5940a`](https://github.com/mizorewww/laya-mlx/blob/ca5940aa9286dbbdfaacbecbdb7b337295ad36a2/README.md)。

上游能力可能在这些快照之后变化。这是文档对照，JEV MLX 尚未与上述项目运行
共同基准测试。
