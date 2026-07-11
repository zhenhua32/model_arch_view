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
- 对 LLM 支持三种 Decoder Stack 视图：汇总、单层 Block（含 Q/K/V/Attention Output）、重复 N 层摘要
- 右侧查看节点的关键配置字段、输入 shape、输出 shape、推导公式和上下游跳转
- 支持导出当前模型视图的 JSON 数据和 SVG 图

## Shape 说明

页面展示的是基于配置文件和用户输入参数推导出的典型 shape，不是逐算子运行时真实张量。原因是当前仓库只保存了配置类文件，没有真实执行图，也没有运行时 trace。

不同类型模型的 shape 推导方式如下：

- LLM：根据 hidden_size、num_hidden_layers、attention heads、MoE 配置推导 token 流形状
- 多模态：根据 patch_size、merge_size、vision_config、processor_config 推导图像或视频 token 数
- Diffusers：根据 VAE 下采样倍率、transformer patch_size、scheduler 步数推导 latent token 流

## 已知限制

- 对未适配的自定义模型目录，只会展示摘要而不是完整图结构
- 某些模型的真实视觉 token 合并策略可能比配置文件更复杂，因此图中的 token 数是近似值
- 是否必须输入条件图像，部分 diffusion pipeline 只能根据目录结构做推断，页面会给出提示

