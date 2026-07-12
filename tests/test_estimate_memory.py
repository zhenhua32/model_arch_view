"""Tests for estimate_memory_footprint: precision scaling, KV linearity,
monotonicity, and GPU-count arithmetic.
"""
from __future__ import annotations

import math

import pytest

import serve_model_arch as s


def base_metrics(**over):
    m = {
        "total_params": 1_000_000_000,
        "active_params": 1_000_000_000,
        "kv_bytes_per_token": 4096,
        "hidden_size": 4096,
        "num_layers": 32,
    }
    m.update(over)
    return m


def test_precision_scales_weights():
    bf16 = s.estimate_memory_footprint(base_metrics(), "bf16", 1, 2048)
    fp8 = s.estimate_memory_footprint(base_metrics(), "fp8", 1, 2048)
    int4 = s.estimate_memory_footprint(base_metrics(), "int4", 1, 2048)
    assert fp8["weights_bytes"] == pytest.approx(bf16["weights_bytes"] / 2)
    assert int4["weights_bytes"] == pytest.approx(bf16["weights_bytes"] / 4)


def test_kv_linear_in_seq_and_batch():
    a = s.estimate_memory_footprint(base_metrics(), "bf16", 1, 2048)
    b = s.estimate_memory_footprint(base_metrics(), "bf16", 1, 4096)
    c = s.estimate_memory_footprint(base_metrics(), "bf16", 2, 2048)
    assert b["kv_bytes"] == pytest.approx(a["kv_bytes"] * 2)
    assert c["kv_bytes"] == pytest.approx(a["kv_bytes"] * 2)


def test_kv_bytes_exact():
    m = base_metrics(kv_bytes_per_token=4096)
    r = s.estimate_memory_footprint(m, "bf16", 3, 1000)
    assert r["kv_bytes"] == 4096 * 1000 * 3


def test_total_is_sum_of_parts():
    r = s.estimate_memory_footprint(base_metrics(), "bf16", 2, 4096)
    assert r["total_bytes"] == r["weights_bytes"] + r["kv_bytes"] + r["activation_bytes"]


def test_total_monotonic_in_seq():
    small = s.estimate_memory_footprint(base_metrics(), "bf16", 1, 1024)
    large = s.estimate_memory_footprint(base_metrics(), "bf16", 1, 8192)
    assert large["total_bytes"] > small["total_bytes"]


def test_gpu_fit_counts():
    r = s.estimate_memory_footprint(base_metrics(), "bf16", 1, 2048)
    by_name = {g["name"]: g["count"] for g in r["gpu_fit"]}
    assert {g["name"] for g in s.GPU_REFERENCE} == set(by_name)
    for gpu in s.GPU_REFERENCE:
        usable = gpu["mem_gb"] * s._VRAM_USABLE
        assert by_name[gpu["name"]] == math.ceil(r["total_gb"] / usable)


def test_bigger_gpu_needs_fewer_cards():
    # 400 GB model: small cards need more units than big cards.
    r = s.estimate_memory_footprint(base_metrics(total_params=200_000_000_000), "bf16", 1, 2048)
    counts = {g["name"]: g["count"] for g in r["gpu_fit"]}
    assert counts["RTX 4090"] > counts["H200 141G"]


def test_unknown_precision_falls_back_to_bf16():
    r = s.estimate_memory_footprint(base_metrics(), "float3", 1, 2048)
    assert r["precision"] == "bf16"


def test_sliding_layers_cap_kv_cache():
    metrics = base_metrics(
        kv_bytes_per_token=28 * 2048,
        kv_bytes_per_token_per_layer=2048,
        num_layers=28,
        full_attention_layers=7,
        sliding_attention_layers=21,
        sliding_window=1024,
    )
    result = s.estimate_memory_footprint(metrics, "bf16", 1, 131072)
    expected = 2048 * (7 * 131072 + 21 * 1024)
    assert result["kv_bytes"] == expected


def test_activation_workspace_is_reused_across_layers():
    one_layer = s.estimate_memory_footprint(base_metrics(num_layers=1), "bf16", 1, 2048)
    many_layers = s.estimate_memory_footprint(base_metrics(num_layers=80), "bf16", 1, 2048)
    assert one_layer["activation_bytes"] == many_layers["activation_bytes"]
