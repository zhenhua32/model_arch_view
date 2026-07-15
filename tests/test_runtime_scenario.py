from __future__ import annotations

import math

import pytest

import serve_model_arch as s


def scenario_metrics(**overrides):
    metrics = {
        "total_params": 7_000_000_000,
        "active_params": 7_000_000_000,
        "hidden_size": 4096,
        "num_layers": 32,
        "linear_flops_per_token": 14_000_000_000,
        "kv_bytes_per_token": 524_288,
    }
    metrics.update(overrides)
    return metrics


@pytest.mark.parametrize("workload", s.WORKLOAD_OPTIONS)
def test_runtime_scenario_returns_finite_metrics(workload):
    result = s.estimate_runtime_scenario(
        scenario_metrics(),
        workload=workload,
        gpu_name="H100 80G",
        gpu_count=2,
    )
    assert result["workload"] == workload
    assert result["memory"]["total_bytes"] > 0
    assert result["memory"]["per_gpu_bytes"] == pytest.approx(result["memory"]["total_bytes"] / 2)
    assert result["performance"]["primary_value"] > 0
    assert 0.6 <= result["parallel_efficiency"] <= 1


def test_training_memory_includes_optimizer_and_model_states():
    metrics = scenario_metrics(total_params=1_000_000_000, hidden_size=1024, num_layers=8)
    result = s.estimate_runtime_scenario(metrics, workload="training", batch=2, seq_len=1024)
    memory = result["memory"]
    expected_activation = 2 * 1024 * 1024 * 8 * s._TRAIN_ACT_FACTOR * 2
    assert memory["weights_bytes"] == 2_000_000_000
    assert memory["gradients_bytes"] == 2_000_000_000
    assert memory["master_weights_bytes"] == 4_000_000_000
    assert memory["optimizer_bytes"] == 8_000_000_000
    assert memory["activation_bytes"] == expected_activation
    assert memory["total_bytes"] == sum(
        memory[key]
        for key in (
            "weights_bytes",
            "kv_bytes",
            "activation_bytes",
            "gradients_bytes",
            "master_weights_bytes",
            "optimizer_bytes",
        )
    )


def test_low_bit_training_uses_bf16_compute_and_trainable_weights():
    bf16 = s.estimate_runtime_scenario(scenario_metrics(), workload="training", precision="bf16")
    int4 = s.estimate_runtime_scenario(scenario_metrics(), workload="training", precision="int4")
    assert int4["memory"]["weights_bytes"] == bf16["memory"]["weights_bytes"]
    assert int4["performance"]["training_compute_precision"] == "bf16"
    assert int4["gpu"]["compute_tops"] == bf16["gpu"]["compute_tops"]
    assert int4["performance"]["training_tps"] == pytest.approx(bf16["performance"]["training_tps"])


def test_gpu_capacity_and_minimum_count_are_consistent():
    result = s.estimate_runtime_scenario(
        scenario_metrics(total_params=70_000_000_000, active_params=70_000_000_000),
        gpu_name="RTX 4090",
        gpu_count=1,
    )
    usable_per_gpu = 24 * s._VRAM_USABLE * 1024**3
    assert result["memory"]["fits"] is False
    assert result["memory"]["minimum_gpu_count"] == math.ceil(result["memory"]["total_bytes"] / usable_per_gpu)


def test_auto_parallelism_uses_expert_for_large_moe_cluster():
    result = s.estimate_runtime_scenario(
        scenario_metrics(routed_experts_active=1_000_000_000),
        gpu_count=8,
        parallelism="auto",
    )
    assert result["parallelism"] == "expert"


def test_expert_parallelism_falls_back_for_dense_model():
    result = s.estimate_runtime_scenario(scenario_metrics(), gpu_count=4, parallelism="expert")
    assert result["requested_parallelism"] == "expert"
    assert result["parallelism"] == "tensor"


def test_pipeline_parallelism_falls_back_when_stages_exceed_layers():
    result = s.estimate_runtime_scenario(
        scenario_metrics(num_layers=2), gpu_count=4, parallelism="pipeline"
    )
    assert result["requested_parallelism"] == "pipeline"
    assert result["parallelism"] == "tensor"


def test_real_payload_exposes_scenario_controls_and_metrics():
    payload = s.build_model_payload(
        "Qwen__Qwen2.5-7B-Instruct",
        {
            "workload": ["training"],
            "gpu": ["H200 141G"],
            "gpu_count": ["4"],
            "parallelism": ["pipeline"],
        },
    )
    controls = {control["name"]: control for control in payload["controls"]}
    scenario = payload["metrics"]["scenario"]

    assert {"workload", "gpu", "gpu_count", "parallelism"}.issubset(controls)
    assert payload["parameters"]["workload"] == "training"
    assert scenario["gpu"]["name"] == "H200 141G"
    assert scenario["gpu_count"] == 4
    assert scenario["parallelism"] == "pipeline"
    assert scenario["memory"]["gradients_bytes"] > 0
    assert any(item["id"] == "deployment-scenario" for item in payload["audit"]["evidence"])


def test_embedding_payload_rejects_decode_workload():
    payload = s.build_model_payload("BAAI__bge-m3", {"workload": ["decode"]})
    assert payload["parameters"]["workload"] == "prefill"
    workload_control = next(control for control in payload["controls"] if control["name"] == "workload")
    assert workload_control["options"] == ["prefill", "training"]
