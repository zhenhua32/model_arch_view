# 模型架构可视化

一个本地网页程序，用来浏览 model_configs 下的模型目录，并把配置文件归一化为可读的架构图、输入输出流向和典型 shape。

当前版本支持三类模型：

- LLM
- 多模态模型
- Diffusers / diffusion pipeline

缺少可计算架构字段的纯元数据仓库、工作流包和 checkpoint 清单会标记为 `unknown`，不会伪造 LLM 参数或 shape。模型卡明确声明 `base_model` 且仓库内存在对应基础配置时，LoRA、重打包或 GGUF 目录会继承基础模型的架构 shape，并明确提示继承来源。

## 启动

在仓库根目录运行：

```powershell
pip install requests modelscope
python serve_model_arch.py
```

默认地址：

```text
http://127.0.0.1:8000
```

也可以自定义端口：

```powershell
python serve_model_arch.py --port 8123
```

## 页面功能

- 自动扫描 model_configs 下的一级模型目录
- 搜索本地不存在的模型时自动查询 ModelScope；点击远端结果后只下载配置、词表和说明文件到 `model_configs`，不会下载权重或模型代码
- 左侧选择模型并调整 batch、token 长度、分辨率、帧数、推理步数等参数
- 中间查看架构图与数据流向，并在选中节点后高亮其上下游路径
- 对 LLM 支持三种 Decoder Stack 视图：汇总、单层 Block（含 Q/K/V、RoPE、QK Score、Causal Mask、Sliding Window Mask、Softmax、Weighted V、Output Projection）、重复 N 层摘要；repeat 视图会额外给出首层 / 中层 / 末层的 mask 摘要
- 右侧查看节点的关键配置字段、输入 shape、输出 shape、推导公式和上下游跳转
- “部署估算”页按 Decode、Prefill 或全量训练场景展示指定 GPU 集群的显存构成、容量下界、并行策略和理论吞吐
- “专项分析”页汇总 LLM 的注意力类型、GQA / MLA、MoE 路由、激活参数和输出头；其他模型展示泳道、节点类型与主计算路径
- “计算审计”页展示可信度、配置继承、字段来源、结构化诊断、公式输入与 checkpoint 真值对比
- “A/B 对比”页按结构、计算容量和当前部署场景分组比较，并标记两端运行参数是否被模型边界校正
- 架构图支持节点固定、父级折叠，并按模型保存层级、缩放、选择和折叠状态
- “复制链接”会编码模型、运行参数、当前页签和图状态，打开链接可还原同一分析现场
- 支持导出当前模型视图的 JSON 数据、SVG 图和 Markdown 审计报告

访问私有或受限模型时，可在启动前设置 ModelScope token：

```powershell
$env:MODELSCOPE_API_TOKEN = "ms-xxxxxxxx"
python serve_model_arch.py
```

## 计算审计

- 每个模型 payload 都包含版本化的 `audit` 数据，覆盖置信度、诊断、证据链、配置来源和 checkpoint 对比。
- LLM 核心字段会标记为直接读取、基础模型继承、公式推导或运行参数，并保留具体文件与字段路径。
- BF16 / FP16 safetensors index 可作为参数量代理真值；量化或混合精度 checkpoint 只比较存储体积，不伪装成精确参数数目。
- 参数分解、激活参数、KV cache、FLOPs 和显存分别记录公式、输入、结果及假设边界。
- 未适配架构、关键字段缺失、别名冲突、层型长度不一致和参数分解不闭合会生成带稳定代码的诊断项。

## Shape 说明

页面展示的是基于配置文件和用户输入参数推导出的典型 shape，不是逐算子运行时真实张量。原因是当前仓库只保存了配置类文件，没有真实执行图，也没有运行时 trace。

不同类型模型的 shape 推导方式如下：

- LLM：根据 hidden_size、num_hidden_layers、attention heads、MoE 配置推导 token 流形状
- 多模态：优先使用显式 soft-token 预算，否则根据 patch_size、merge_size、vision_config、processor_config 推导所选图像、视频或音频分支；ASR、SAM 视频分割、HY-World 3D 世界模型及 MOSS-TTS、CosyVoice、IndexTTS、VoxCPM 等语音合成架构使用专用计算图
- Diffusers：根据 VAE 下采样倍率、transformer patch_size、scheduler 步数推导 latent token 流；没有标准 `model_index.json` 时，也会读取 Wan/3D DiT 等数值 transformer 配置

## 估算口径

- LLM 参数量区分 gated / 非 gated FFN、稠密层、路由专家、共享专家和可选 MTP 辅助层。
- Decode FLOPs 包含线性层以及当前上下文长度对应的 QK / AV 注意力计算。
- KV cache 和 decode 带宽会按 full / sliding attention 层分别应用上下文窗口。
- DeepSeek V4 按共享 K=V、q/grouped-o 低秩和 CSA/HCA 压缩率计算；CSA/DSA 同时计入低维 indexer 扫描与 top-k 主注意力，openPangu DSA 缓存按完整序列计算。
- 原生混合精度模型会分别计算 FP4 路由专家与 FP8 主干的激活权重读取；显存优先采用 safetensors index 的实际 checkpoint 字节数。
- 显存中的激活项按可复用的单层 inference workspace 粗估，不按层数重复累计。
- GPU 卡数仍只是容量下界；真实部署还受并行切分、通信和框架常驻显存影响。
- 多卡场景先聚合理想算力与带宽，再乘按卡数和并行策略估计的效率；该值不替代真实拓扑 benchmark。
- 全量训练显存包含 BF16 权重与梯度、FP32 master weight、Adam FP32 m/v 和按层累计的激活粗估，暂不建模 ZeRO、FSDP、重计算与 offload。
- Diffusers 视频模型的 token 数包含 VAE 时间压缩与 3D transformer patch。
- 多码本 delay-pattern TTS 会计算码本错位后的解码步；单码流 TTS 不再虚构 RVQ 并行头，只有配置明确给出帧率、mel 比例和 hop 时才换算波形样本数。
- HY-World 同时计算 HY-Pano 的 MoE 参数、VAE latent 与环形融合宽度，以及 WorldMirror 的多视图 patch token 和过滤前 Gaussian 数量。
- Nemotron 流式 ASR 按模型卡公开的 80ms 基础帧、右上下文和 56 帧左缓存计算各档 chunk；未公开的前端和 RNN-T 维度保留为符号值。

## 已知限制

- 对未适配的自定义模型目录，只会展示摘要而不是完整图结构
- 未声明固定 token 数的动态视觉切片策略仍可能比配置文件更复杂，因此页面会同时展示 patch 网格值与有效 token 值
- 是否必须输入条件图像，部分 diffusion pipeline 只能根据目录结构做推断，页面会给出提示
- checkpoint 显存包含索引声明的量化元数据与辅助层，但不同推理引擎的临时 buffer、通信开销和 kernel 效率不包含在理论上限中

