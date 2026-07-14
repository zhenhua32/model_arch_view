"""White-box unit tests for the estimation primitives in serve_model_arch.

These use small synthetic configs so every arithmetic path (MLA, GQA/MHA, MoE
dense-layer split, tied embeddings, alias resolution, list-valued patch_size) is
checked against a hand-computed expected value -- independent of any real model.
"""
from __future__ import annotations

import math

import pytest

import serve_model_arch as s


def test_read_json_file_rejects_non_mapping_roots(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    assert s.read_json_file(path) == {}


# --------------------------------------------------------------------------- #
# scalar_int -- video patch_size may be a list [t, h, w]
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        (2, 2),
        ([1, 2, 2], 2),          # video patch -> spatial (last) dim
        ((16,), 16),
        ([], 1),                 # empty -> default
        (None, 1),               # missing -> default
        (0, 1),                  # non-positive -> default
        ("not-a-number", 1),     # garbage -> default
    ],
)
def test_scalar_int(value, expected):
    assert s.scalar_int(value, 1) == expected


# --------------------------------------------------------------------------- #
# Dense (non-MoE) GQA model: hand-computed parameter count
# --------------------------------------------------------------------------- #
def test_dense_gqa_param_count():
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,      # GQA
        "head_dim": 64,                # num_heads*head_dim = 1024 == hidden
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
    }
    m = s.estimate_llm_metrics(s.parse_llm_dims(cfg))

    H, L = 1024, 2
    q_dim = 16 * 64          # 1024
    kv_dim = 4 * 64          # 256
    attn = H * q_dim + H * kv_dim + H * kv_dim + q_dim * H
    ffn = 3 * H * 4096
    embed = H * 32000
    expected_total = attn * L + ffn * L + embed * 2          # untied
    expected_active = attn * L + ffn * L + embed             # lm_head once

    assert m["attn_params_per_layer"] == attn
    assert m["total_params"] == expected_total
    assert m["active_params"] == expected_active
    assert m["gflops_per_token"] == pytest.approx(2 * expected_active / 1e9)


def test_qdim_differs_from_hidden():
    """Qwen3-style: num_heads*head_dim != hidden_size must be respected."""
    cfg = {
        "hidden_size": 5120,
        "num_hidden_layers": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,               # 64*128 = 8192 != 5120
        "intermediate_size": 25600,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    m = s.estimate_llm_metrics(s.parse_llm_dims(cfg))
    H = 5120
    q_dim = 64 * 128       # 8192
    kv_dim = 8 * 128       # 1024
    attn = H * q_dim + H * kv_dim + H * kv_dim + q_dim * H
    assert m["attn_params_per_layer"] == attn
    # Sanity: the naive 2*H*H assumption would be wrong / smaller.
    assert attn != 2 * H * H


# --------------------------------------------------------------------------- #
# tie_word_embeddings: embedding counted once vs twice
# --------------------------------------------------------------------------- #
def test_tied_embeddings_counted_once():
    base = {
        "hidden_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 2048,
        "vocab_size": 50000,
    }
    tied = s.estimate_llm_metrics(s.parse_llm_dims({**base, "tie_word_embeddings": True}))
    untied = s.estimate_llm_metrics(s.parse_llm_dims({**base, "tie_word_embeddings": False}))
    embed = 512 * 50000
    assert untied["total_params"] - tied["total_params"] == embed
    # active traverses lm_head once in both cases -> identical active counts
    assert tied["active_params"] == untied["active_params"]


# --------------------------------------------------------------------------- #
# MoE: routed experts use moe_intermediate_size; first_k_dense are dense layers
# --------------------------------------------------------------------------- #
def test_moe_expert_dim_and_dense_split():
    cfg = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "intermediate_size": 8192,        # dense FFN dim
        "moe_intermediate_size": 1024,    # routed/shared expert dim (much smaller)
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,       # layer 0 is dense, layers 1-3 are MoE
        "vocab_size": 10000,
        "tie_word_embeddings": False,
    }
    m = s.estimate_llm_metrics(s.parse_llm_dims(cfg))

    H = 2048
    dense_ffn = 3 * H * 8192
    expert = 3 * H * 1024
    routed_total = expert * 8
    routed_active = expert * 2
    shared = expert * 1
    n_dense, n_moe = 1, 3
    ffn_total = n_dense * dense_ffn + n_moe * (routed_total + shared)
    ffn_active = n_dense * dense_ffn + n_moe * (routed_active + shared)

    assert m["ffn_total"] == ffn_total
    assert m["ffn_active"] == ffn_active
    # active must be far smaller than total for a sparse MoE
    assert m["active_params"] < m["total_params"]


def test_moe_uses_moe_intermediate_not_dense():
    """Regression: experts must NOT use the dense intermediate_size."""
    common = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "n_routed_experts": 16,
        "num_experts_per_tok": 2,
        "vocab_size": 10000,
    }
    small = s.estimate_llm_metrics(s.parse_llm_dims({**common, "moe_intermediate_size": 1024}))
    big = s.estimate_llm_metrics(s.parse_llm_dims({**common, "moe_intermediate_size": 4096}))
    # Larger expert dim -> strictly more parameters.
    assert big["total_params"] > small["total_params"]


@pytest.mark.parametrize("alias", ["n_routed_experts", "num_experts", "num_local_experts"])
def test_expert_count_aliases(alias):
    """MiniMax uses num_local_experts; others use n_routed_experts/num_experts."""
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 2048,
        "num_experts_per_tok": 2,
        "vocab_size": 10000,
        alias: 32,
    }
    dims = s.parse_llm_dims(cfg)
    assert dims["num_experts"] == 32


def test_longcat_style_aliases():
    """LongCat uses num_layers + expert_ffn_hidden_size instead of the usual keys."""
    cfg = {
        "hidden_size": 2048,
        "num_layers": 4,                      # not num_hidden_layers
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "expert_ffn_hidden_size": 1024,       # not moe_intermediate_size
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 10000,
    }
    dims = s.parse_llm_dims(cfg)
    assert dims["num_layers"] == 4
    assert dims["moe_ffn_hidden"] == 1024
    m = s.estimate_llm_metrics(dims)
    assert m is not None and m["total_params"] > 0


def test_attention_layer_types_are_exclusive_and_missing_entries_are_full():
    counts = s._attention_layer_counts(["linear_full", "sliding_attention"], 3, True)
    assert counts == (1, 1, 1)


def test_sparse_layers_take_precedence_over_overlapping_sliding_layers():
    dims = s.parse_llm_dims(
        {
            "num_hidden_layers": 4,
            "dsa_layers": [0, 1],
            "swa_layers": [1, 2],
            "index_topk": 128,
        }
    )
    assert dims["sparse_attention_layers"] == 2
    assert dims["sliding_attention_layers"] == 1
    assert dims["full_attention_layers"] == 1


def test_deepseek_v4_compression_ratios_are_padded_to_layer_count():
    dims = s.parse_llm_dims(
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 4,
            "compress_ratios": [4],
        }
    )
    assert dims["compress_ratios"] == [4, 0, 0, 0]


def test_deepseek_v4_context_keeps_compressed_tokens_without_index_topk():
    metrics = {
        "is_deepseek_v4": True,
        "compress_ratios": [4],
        "sliding_window": 0,
        "head_dim": 128,
        "index_topk": 0,
    }
    assert s._context_layer_tokens(metrics, 16) == 20


def test_active_expert_count_cannot_exceed_total_experts():
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "head_dim": 128,
        "moe_intermediate_size": 512,
        "num_experts": 2,
        "num_experts_per_tok": 4,
        "cli_factor": 2,
        "vocab_size": 10000,
    }
    metrics = s.estimate_llm_metrics(s.parse_llm_dims(cfg))
    assert metrics["effective_experts_per_tok"] == 2
    assert metrics["ffn_active"] <= metrics["ffn_total"]


# --------------------------------------------------------------------------- #
# MLA attention + latent KV cache
# --------------------------------------------------------------------------- #
def test_mla_attention_and_kv_cache():
    cfg = {
        "hidden_size": 6144,
        "num_hidden_layers": 4,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 154880,
        "tie_word_embeddings": False,
    }
    dims = s.parse_llm_dims(cfg)
    assert dims["is_mla"] is True
    m = s.estimate_llm_metrics(dims)

    H, L = 6144, 4
    num_heads = 64
    qk_head = 192 + 64
    q_params = H * 2048 + 2048 * num_heads * qk_head
    kv_down = H * (512 + 64)
    kv_up = 512 * num_heads * (192 + 256)
    o_params = num_heads * 256 * H
    attn = q_params + kv_down + kv_up + o_params
    assert m["attn_params_per_layer"] == attn

    # MLA caches only the compressed latent + rope key: (kv_lora + rope) per layer.
    kv_bytes = L * (512 + 64) * 2
    expected_mb = kv_bytes * 1024 / (1024 * 1024)
    assert m["kv_cache_mb_per_1k"] == pytest.approx(expected_mb)


def test_mla_kv_cache_far_smaller_than_full_gqa():
    """The MLA latent cache must be dramatically smaller than a naive full-KV cache."""
    cfg = {
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 154880,
    }
    m = s.estimate_llm_metrics(s.parse_llm_dims(cfg))
    # Naive full GQA cache (64 heads * 256 head_dim) would be enormous; the MLA
    # latent cache stores only 512+64 per layer -> well under 200 MB / 1K tokens.
    assert m["kv_cache_mb_per_1k"] < 200


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def test_missing_dims_returns_none():
    assert s.estimate_llm_metrics(s.parse_llm_dims({})) is None
    assert s.estimate_llm_metrics(s.parse_llm_dims({"hidden_size": 1024})) is None


def test_flops_is_twice_active_params():
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 2048,
        "vocab_size": 10000,
    }
    m = s.estimate_llm_metrics(s.parse_llm_dims(cfg))
    assert m["gflops_per_token"] == pytest.approx(2 * m["active_params"] / 1e9)
    assert not math.isnan(m["gflops_per_token"])


def test_shared_expert_alias_and_width_are_respected():
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "head_dim": 128,
        "moe_intermediate_size": 512,
        "moe_shared_expert_intermediate_size": 768,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "vocab_size": 10000,
    }
    dims = s.parse_llm_dims(cfg)
    metrics = s.estimate_llm_metrics(dims)
    assert dims["n_shared_experts"] == 1
    assert dims["shared_ffn_hidden"] == 768
    assert metrics["shared_expert_params"] == 3 * 1024 * 768
    assert metrics["breakdown"]["shared_experts"] == 4 * 3 * 1024 * 768


def test_longcat_aliases_enable_topk_cli_and_mtp():
    cfg = {
        "hidden_size": 1024,
        "num_layers": 4,
        "num_attention_heads": 8,
        "head_dim": 128,
        "expert_ffn_hidden_size": 512,
        "n_routed_experts": 16,
        "moe_topk": 3,
        "cli_factor": 2,
        "mtp_num_layers": 2,
        "vocab_size": 10000,
    }
    dims = s.parse_llm_dims(cfg)
    metrics = s.estimate_llm_metrics(dims)
    assert dims["experts_per_tok"] == 3
    assert metrics["effective_experts_per_tok"] == 6
    assert metrics["mtp_layers"] == 2
    assert metrics["breakdown"]["mtp"] == metrics["mtp_params"] > 0


def test_encoder_uses_two_projection_ffn_without_lm_head_or_kv():
    cfg = {
        "architectures": ["XLMRobertaModel"],
        "model_type": "xlm-roberta",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 8194,
        "type_vocab_size": 1,
        "vocab_size": 250002,
    }
    dims = s.parse_llm_dims(cfg)
    metrics = s.estimate_llm_metrics(dims)
    assert dims["ffn_projection_count"] == 2
    assert dims["has_output_head"] is False
    assert metrics["output_head_params"] == 0
    assert metrics["kv_cache_mb_per_1k"] is None
    assert metrics["total_params"] == 566_383_616
