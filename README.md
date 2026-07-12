# 模型架构可视化

一个本地网页程序，用来浏览 model_configs 下的模型目录，并把配置文件归一化为可读的架构图、输入输出流向和典型 shape。

当前版本支持三类模型：

- LLM
- 多模态模型
- Diffusers / diffusion pipeline

## 启动

在仓库根目录运行：

```powershell
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
- 左侧选择模型并调整 batch、token 长度、分辨率、帧数、推理步数等参数
- 中间查看架构图与数据流向，并在选中节点后高亮其上下游路径
- 对 LLM 支持三种 Decoder Stack 视图：汇总、单层 Block（含 Q/K/V、RoPE、QK Score、Causal Mask、Sliding Window Mask、Softmax、Weighted V、Output Projection）、重复 N 层摘要；repeat 视图会额外给出首层 / 中层 / 末层的 mask 摘要
- 右侧查看节点的关键配置字段、输入 shape、输出 shape、推导公式和上下游跳转
- 支持导出当前模型视图的 JSON 数据和 SVG 图

## Shape 说明

页面展示的是基于配置文件和用户输入参数推导出的典型 shape，不是逐算子运行时真实张量。原因是当前仓库只保存了配置类文件，没有真实执行图，也没有运行时 trace。

不同类型模型的 shape 推导方式如下：

- LLM：根据 hidden_size、num_hidden_layers、attention heads、MoE 配置推导 token 流形状
- 多模态：优先使用显式 soft-token 预算，否则根据 patch_size、merge_size、vision_config、processor_config 推导所选图像、视频或音频分支；ASR、SAM 视频分割和 delay-pattern TTS 使用专用计算图
- Diffusers：根据 VAE 下采样倍率、transformer patch_size、scheduler 步数推导 latent token 流

## 估算口径

- LLM 参数量区分 gated / 非 gated FFN、稠密层、路由专家、共享专家和可选 MTP 辅助层。
- Decode FLOPs 包含线性层以及当前上下文长度对应的 QK / AV 注意力计算。
- KV cache 和 decode 带宽会按 full / sliding attention 层分别应用上下文窗口。
- DeepSeek V4 按共享 K=V、q/grouped-o 低秩和 CSA/HCA 压缩率计算；CSA/DSA 同时计入低维 indexer 扫描与 top-k 主注意力，openPangu DSA 缓存按完整序列计算。
- 原生混合精度模型会分别计算 FP4 路由专家与 FP8 主干的激活权重读取；显存优先采用 safetensors index 的实际 checkpoint 字节数。
- 显存中的激活项按可复用的单层 inference workspace 粗估，不按层数重复累计。
- GPU 卡数仍只是容量下界；真实部署还受并行切分、通信和框架常驻显存影响。
- Diffusers 视频模型的 token 数包含 VAE 时间压缩与 3D transformer patch。

## 已知限制

- 对未适配的自定义模型目录，只会展示摘要而不是完整图结构
- 未声明固定 token 数的动态视觉切片策略仍可能比配置文件更复杂，因此页面会同时展示 patch 网格值与有效 token 值
- 是否必须输入条件图像，部分 diffusion pipeline 只能根据目录结构做推断，页面会给出提示
- checkpoint 显存包含索引声明的量化元数据与辅助层，但不同推理引擎的临时 buffer、通信开销和 kernel 效率不包含在理论上限中

