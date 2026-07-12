"""Parameter-breakdown tests: sum(breakdown) must equal total_params, and the
component split must be structurally correct for MoE vs dense models.
"""
from __future__ import annotations

import pytest

import serve_model_arch as s

from conftest import all_model_dirs


def _metrics_for(model_dir):
    cfg = s.primary_config(model_dir)
    pc = s.read_json_file(model_dir / "params.json") or {}
    return s.estimate_llm_metrics(s.parse_llm_dims(cfg, pc))


def _llm_dirs_with_metrics():
    out = []
    for d in all_model_dirs():
        if s.classify_model_dir(d) != "llm":
            continue
        m = _metrics_for(d)
        if m:
            out.append((d.name, m))
    return out


LLM_METRICS = _llm_dirs_with_metrics()


@pytest.mark.parametrize("name,m", LLM_METRICS, ids=[n for n, _ in LLM_METRICS])
def test_breakdown_sums_to_total(name, m):
    """All reported components must exactly reconstruct total_params."""
    bd = m["breakdown"]
    assert set(bd) == {"attention", "routed_experts", "shared_experts", "dense_ffn", "embedding", "mtp"}
    assert sum(bd.values()) == m["total_params"]
    assert all(v >= 0 for v in bd.values())


def test_dense_model_has_no_experts():
    cfg = {
        "hidden_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 16,
        "num_key_value_heads": 16, "head_dim": 64, "intermediate_size": 4096,
        "vocab_size": 32000, "tie_word_embeddings": False,
    }
    bd = s.estimate_llm_metrics(s.parse_llm_dims(cfg))["breakdown"]
    assert bd["routed_experts"] == 0
    assert bd["shared_experts"] == 0
    assert bd["dense_ffn"] > 0


def test_moe_routed_experts_dominate():
    """A wide MoE with many experts should have routed_experts as the largest part."""
    cfg = {
        "hidden_size": 2048, "num_hidden_layers": 12, "num_attention_heads": 16,
        "num_key_value_heads": 16, "head_dim": 128, "intermediate_size": 8192,
        "moe_intermediate_size": 1408, "n_routed_experts": 128, "num_experts_per_tok": 8,
        "n_shared_experts": 1, "first_k_dense_replace": 1, "vocab_size": 100000,
    }
    m = s.estimate_llm_metrics(s.parse_llm_dims(cfg))
    bd = m["breakdown"]
    assert bd["routed_experts"] == max(bd.values())
    assert bd["shared_experts"] > 0
    assert bd["dense_ffn"] > 0  # first_k_dense=1 dense layer
    assert m["active_params"] < m["total_params"]
