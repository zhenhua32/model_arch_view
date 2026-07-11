"""Golden-value regression tests for a curated set of anchor models.

These freeze the exact estimated metrics for well-understood models so any future
change to the estimation logic that shifts a number is caught immediately. Values
were verified against published architectures and safetensors byte counts during
the GLM-5.2-FP8 audit and the full-corpus validation sweep.

If an intentional formula change moves these numbers, update the expected values
here in the same commit (and record why in the audit notes).
"""
from __future__ import annotations

import pytest

# model_id -> expected metrics
#   total  : total parameter count
#   active : active (per-token) parameter count
#   kv     : KV cache MiB per 1K tokens per batch element (rounded to .1)
#   gflops : inference GFLOPs per token (rounded to .1)
GOLDEN = {
    # --- MLA + MoE (DeepSeek/GLM/LongCat/Hy3 family) ---
    "ZhipuAI__GLM-5.2-FP8": dict(total=743061061632, active=39982989312, kv=87.8, gflops=80.0),
    "ZhipuAI__GLM-5.2": dict(total=743061061632, active=39982989312, kv=87.8, gflops=80.0),
    "deepseek-ai__DeepSeek-V4-Pro": dict(total=1592836423680, active=68691623936, kv=68.6, gflops=137.4),
    "meituan-longcat__LongCat-2.0-FP8": dict(total=1475806756864, active=7498366976, kv=42.8, gflops=15.0),
    "Tencent-Hunyuan__Hy3": dict(total=293479645184, active=18626904064, kv=320.0, gflops=37.3),
    # --- Standard GQA MoE (num_local_experts alias) ---
    "MiniMax__MiniMax-M2.7": dict(total=228640161792, active=10366353408, kv=248.0, gflops=20.7),
    # --- Dense GQA (bf16, exact vs checkpoint) ---
    "Qwen__Qwen3-8B": dict(total=8190427136, active=7568097280, kv=144.0, gflops=15.1),
    "Qwen__Qwen3-32B": dict(total=32761446400, active=31983534080, kv=256.0, gflops=64.0),
    "Qwen__Qwen2.5-7B-Instruct": dict(total=7615283200, active=7070285824, kv=56.0, gflops=14.1),
    "WeiboAI__VibeThinker-3B": dict(total=3085697024, active=3085697024, kv=36.0, gflops=6.2),
    "OpenBMB__MiniCPM5-1B": dict(total=1080557568, active=880017408, kv=24.0, gflops=1.8),
    # --- Dense-first MoE ---
    "JetBrains__Mellum2-12B-A2.5B-Thinking": dict(total=12145655808, active=2208301056, kv=56.0, gflops=4.4),
}


@pytest.mark.parametrize("model_id", list(GOLDEN), ids=list(GOLDEN))
def test_golden_metrics(serve, project_root, model_id):
    expected = GOLDEN[model_id]
    model_dir = project_root / "model_configs" / model_id
    assert model_dir.exists(), f"missing anchor model {model_id}"

    cfg = serve.primary_config(model_dir)
    params_cfg = serve.read_json_file(model_dir / "params.json") or {}
    m = serve.estimate_llm_metrics(serve.parse_llm_dims(cfg, params_cfg))
    assert m is not None

    assert m["total_params"] == expected["total"], (
        f"{model_id}: total {m['total_params']} != {expected['total']}"
    )
    assert m["active_params"] == expected["active"], (
        f"{model_id}: active {m['active_params']} != {expected['active']}"
    )
    assert round(m["kv_cache_mb_per_1k"], 1) == expected["kv"], (
        f"{model_id}: kv {round(m['kv_cache_mb_per_1k'], 1)} != {expected['kv']}"
    )
    assert round(m["gflops_per_token"], 1) == expected["gflops"], (
        f"{model_id}: gflops {round(m['gflops_per_token'], 1)} != {expected['gflops']}"
    )


def test_active_less_than_total_for_moe(serve, project_root):
    """Sparse MoE models must have active << total."""
    for model_id in [
        "ZhipuAI__GLM-5.2",
        "deepseek-ai__DeepSeek-V4-Pro",
        "MiniMax__MiniMax-M2.7",
        "meituan-longcat__LongCat-2.0-FP8",
    ]:
        model_dir = project_root / "model_configs" / model_id
        cfg = serve.primary_config(model_dir)
        m = serve.estimate_llm_metrics(serve.parse_llm_dims(cfg))
        assert m["active_params"] < m["total_params"] * 0.5, model_id
