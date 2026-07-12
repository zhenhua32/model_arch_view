"""Tests for estimate_throughput: monotonicity in model size, the compute/
bandwidth min() selection, TTFT linearity, and precision effects.
"""
from __future__ import annotations

import pytest

import serve_model_arch as s


def metrics(active):
    return {"active_params": active, "total_params": active}


def test_returns_one_row_per_gpu():
    rows = s.estimate_throughput(metrics(5_000_000_000), "bf16", 2048)
    assert [r["name"] for r in rows] == [g["name"] for g in s.GPU_REFERENCE]
    for r in rows:
        assert r["decode_tps"] > 0
        assert r["ttft_ms"] > 0
        assert r["bound"] in ("算力", "带宽")


def test_bigger_model_is_slower():
    small = s.estimate_throughput(metrics(1_000_000_000), "bf16", 2048)
    large = s.estimate_throughput(metrics(50_000_000_000), "bf16", 2048)
    for a, b in zip(small, large):
        assert a["decode_tps"] > b["decode_tps"]
        assert b["ttft_ms"] > a["ttft_ms"]


def test_ttft_linear_in_seq_len():
    a = s.estimate_throughput(metrics(2_000_000_000), "bf16", 1024)
    b = s.estimate_throughput(metrics(2_000_000_000), "bf16", 4096)
    for x, y in zip(a, b):
        assert y["ttft_ms"] == pytest.approx(x["ttft_ms"] * 4)


def test_decode_matches_min_of_compute_and_bandwidth():
    active = 3_000_000_000
    rows = s.estimate_throughput(metrics(active), "bf16", 2048)
    by_name = {r["name"]: r for r in rows}
    for gpu in s.GPU_REFERENCE:
        compute = gpu["bf16_tflops"] * 1e12 * s._MFU / (2 * active)
        bw = gpu["bw_gbs"] * 1e9 / (active * s.PRECISION_BYTES["bf16"])
        assert by_name[gpu["name"]]["decode_tps"] == pytest.approx(min(compute, bw))


def test_lower_precision_improves_bandwidth_bound_decode():
    bf16 = s.estimate_throughput(metrics(3_000_000_000), "bf16", 2048)
    int4 = s.estimate_throughput(metrics(3_000_000_000), "int4", 2048)
    # int4 quarters the bytes/param, so bandwidth-bound decode should not be slower.
    for a, b in zip(bf16, int4):
        assert b["decode_tps"] >= a["decode_tps"]


def test_zero_active_returns_empty():
    assert s.estimate_throughput(metrics(0), "bf16", 2048) == []


def transformer_metrics():
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    return s.estimate_llm_metrics(s.parse_llm_dims(cfg))


def test_context_attention_makes_ttft_superlinear():
    m = transformer_metrics()
    short = s.estimate_throughput(m, "bf16", 1024)
    long = s.estimate_throughput(m, "bf16", 4096)
    for a, b in zip(short, long):
        assert b["ttft_ms"] > a["ttft_ms"] * 4


def test_long_context_kv_reads_reduce_decode_ceiling():
    m = transformer_metrics()
    short = s.estimate_throughput(m, "bf16", 1024)
    long = s.estimate_throughput(m, "bf16", 32768)
    for a, b in zip(short, long):
        assert b["decode_tps"] < a["decode_tps"]
        assert b["kv_read_bytes"] > a["kv_read_bytes"]


def test_batch_changes_aggregate_throughput_and_ttft():
    m = transformer_metrics()
    one = s.estimate_throughput(m, "bf16", 2048, batch=1)
    eight = s.estimate_throughput(m, "bf16", 2048, batch=8)
    for a, b in zip(one, eight):
        assert b["decode_tps"] >= a["decode_tps"]
        assert b["ttft_ms"] == pytest.approx(a["ttft_ms"] * 8)


def test_multi_gpu_rows_use_aggregate_ideal_ceiling():
    m = transformer_metrics()
    one = s.estimate_throughput(m, "bf16", 2048, batch=1)
    counts = {gpu["name"]: 2 for gpu in s.GPU_REFERENCE}
    two = s.estimate_throughput(m, "bf16", 2048, batch=1, gpu_counts=counts)
    for a, b in zip(one, two):
        assert b["gpu_count"] == 2
        assert b["compute_tps"] == pytest.approx(a["compute_tps"] * 2)
        assert b["bandwidth_tps"] == pytest.approx(a["bandwidth_tps"] * 2)
        assert b["ttft_ms"] == pytest.approx(a["ttft_ms"] / 2)
