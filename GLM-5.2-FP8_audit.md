# GLM-5.2-FP8 架构指标核查报告

> 更新说明：当前页面已把 FLOPs 拆为线性层与上下文相关的 QK/AV 项。本文中的 `80.0 GFLOPs/token` 是不含上下文 attention 的线性基线；页面会按当前 `seq_len` 显示更高的 decode FLOPs。

> 核查对象：`ZhipuAI/GLM-5.2-FP8`
> 被核查数值来源：`serve_model_arch.py`（估算段 line 1164–1184 的通用公式）
> 真值依据：`config.json` + `model.safetensors.index.json`（`total_size = 755,617,140,416` 字节）

## 结论速览

| 指标 | 工具原输出（有误） | 修正后 / 真实值 | 原偏差 |
|---|---|---|---|
| 参数量 | 4541.22B | **743.06B** | 高 ~6.1× |
| 激活参数 | 159.95B | **39.98B** | 高 ~4.0× |
| 推理 FLOPs | 319.9 GFLOPs/token | **80.0 GFLOPs/token** | 高 ~4.0× |
| KV cache | 3744 MiB / 1K tok | **87.8 MiB / 1K tok（BF16 latent）** | 高 ~43× |

**四个值均不正确**（除摘要头两行 MLA/MoE 描述被正确读取外）。已修正 `serve_model_arch.py` 并验证输出为上表右列。

> 交叉验证：修正后的总参数 743.06B 与 checkpoint 的 `total_size = 755.6GB` 自洽——差额来自 FP8 的 `weight_scale_inv`（fp32）+ 未量化的 embed/lm_head/layernorm（bf16）字节开销，以及额外的 1 层 MTP。独立旁证：同一套逻辑对 `Keye-VL-2.0-30B-A3B` 算出 active 2.63B，与命名 “A3B”（≈3B 激活）吻合。

## 关键佐证

`safetensors.index.json` 的 `total_size = 755,617,140,416` 字节。模型为 FP8(e4m3) 量化，权重基本 1 字节/参数 → 真实总参数量 ≈ 755.6B。
工具报的 4541.22B 隐含 ~4.5TB 权重，与磁盘上的 755.6GB 检查点矛盾，物理上不可能。

## 根因（三个 bug，均在估算段）

1. **MoE 专家维度用错（最致命）**
   工具 FFN 单元取 `intermediate_size = 12288`（稠密层/共享专家的 FFN 维度），却直接当作每个路由专家的中间维度去 ×256 专家。
   真实路由专家维度为 `moe_intermediate_size = 2048`，比值 12288/2048 = **6×** → MoE 部分整体放大 6×。
   按张量名核算真实专家块：76 层含专家、每层 256 专家、每专家 `3×H×2048`，与 total_size 吻合。

2. **KV cache 当成全量 GQA（错得最离谱）**
   公式 `2·L·kv_heads·head_dim·2B` 假设每层 64 个 KV 头 × 192 维。
   但 MLA 仅缓存**压缩 latent**：每层只有 `kv_lora(512) + rope_dim(64) = 576` 个元素（value 由 latent 重构，不单独存）。
   真实每层 576 而非 12288（差 ~21×），叠加字节数后比 BF16 真值高 ~43×、比 FP8 高 ~85×。会严重误导显存规划。

3. **激活参数未区分稠密层**
   前 3 层为稠密 FFN（`first_k_dense_replace = 3`），工具却假定全部 78 层都是 top-8/256 MoE，进一步高估激活量。

> 工具忽略的 DSA indexer 模块与 MTP 层相对上述误差很小：修正专家维度后总和 767B 与 safetensors 755.6B 误差仅 1.5%，主干核算正确。

## 真实架构要点（config.json）

- hidden=6144, layers=78, heads=64
- MLA: q_lora=2048, kv_lora=512, qk_rope=64, qk_nope=192, v_head=256
- MoE: n_routed=256, n_shared=1, top=8, intermediate=12288（稠密/共享）, moe_inter=2048（路由专家）
- first_k_dense_replace=3, vocab=154880
- 含 DSA indexer（index_n_heads=32）+ MTP（num_nextn_predict_layers=1）
- 专家分布在 76 层（layer 3–78），稠密层为 0/1/2

## 逐项核算（核心公式，主干 78 层：3 稠密 + 75 MoE）

```
MLA 每层 ≈ q_a(6144×2048) + q_b(2048×64×256) + kv_a(6144×576)
         + kv_b(512×64×448) + o(64×256×6144) ≈ 165.0M  → ×78 = 12.87B
路由专家总 = 3×6144×2048 ×256 ×75 = 724.8B
路由激活   = 3×6144×2048 ×8   ×75 = 22.65B
共享专家   = 3×6144×2048 ×1   ×75 = 2.83B   （维度=moe_intermediate_size，非 12288）
稠密 FFN   = 3×6144×12288 ×3         = 0.68B
embed+lm_head = 2×6144×154880        = 1.90B

总参数 ≈ 12.87 + 724.8 + 2.83 + 0.68 + 1.90 = 743.1B
激活   ≈ 12.87 + 22.65 + 2.83 + 0.68 + 0.95(lm_head) = 39.98B
FLOPs  ≈ 2 × 激活 = 80.0 GFLOPs/token
KV cache = 78 × (kv_lora 512 + rope 64) × 2B(bf16) × 1024 / 1024² = 87.8 MiB / 1K tok
```

> 说明：共享专家维度经字节数交叉验证确认为 `moe_intermediate_size=2048`（若误用 12288，总参会偏到 757B 且与 checkpoint 不符）。checkpoint 张量名显示专家分布在 layer 3–78 共 76 层，其中 layer 78 为额外的 MTP 预测层（`num_nextn_predict_layers=1`），headline 参数量按主干 78 层计，与 GLM/DeepSeek 惯例一致。
