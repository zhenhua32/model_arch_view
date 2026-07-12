"""Regression checks tied to real configs that exposed estimator bugs."""
from __future__ import annotations

import serve_model_arch as s


def _summary(payload):
    return {item["label"]: item["value"] for item in payload["model"]["summary"]}


def test_hy3_shared_expert_matches_model_card_scale():
    payload = s.build_model_payload("Tencent-Hunyuan__Hy3", {})
    metrics = payload["metrics"]
    assert metrics["total_params"] == 294_970_720_256
    assert metrics["active_params"] == 20_117_979_136
    assert metrics["breakdown"]["shared_experts"] > 0
    assert 294 <= metrics["total_params"] / 1e9 <= 296
    assert 20 <= metrics["active_params"] / 1e9 <= 21


def test_longcat_uses_top12_cli_mtp_and_native_precision():
    payload = s.build_model_payload("meituan-longcat__LongCat-2.0-FP8", {})
    metrics = payload["metrics"]
    assert metrics["effective_experts_per_tok"] == 24
    assert metrics["mtp_layers"] == 3
    assert 1.58e12 <= metrics["total_params"] <= 1.61e12
    assert 45e9 <= metrics["active_params"] <= 55e9
    assert metrics["memory"]["precision"] == "fp8"
    assert metrics["memory"]["weight_source"] == "checkpoint"
    assert metrics["memory"]["weights_bytes"] == 2_051_152_709_632

    int8_payload = s.build_model_payload("meituan-longcat__LongCat-2.0-INT8", {})
    assert int8_payload["metrics"]["memory"]["precision"] == "int8"
    fp4_payload = s.build_model_payload("XiaomiMiMo__MiMo-V2.5-Pro-FP4-DFlash", {})
    assert fp4_payload["metrics"]["memory"]["precision"] == "int4"


def test_bge_is_encoder_without_causal_or_decode_metrics():
    payload = s.build_model_payload("BAAI__bge-m3", {})
    node_ids = {node["id"] for node in payload["graph"]["nodes"]}
    assert "decoder_causal_mask" not in node_ids
    assert "decoder_sliding_mask" not in node_ids
    assert payload["metrics"]["kv_cache_mb_per_1k"] is None
    assert payload["metrics"]["throughput"] == []
    assert "Encoder-only" in payload["model"]["headline"]


def test_real_sliding_window_caps_mellum_kv_but_disabled_qwen_does_not():
    mellum = s.build_model_payload(
        "JetBrains__Mellum2-12B-A2.5B-Thinking",
        {"seq_len": ["131072"]},
    )
    assert mellum["metrics"]["memory"]["kv_bytes"] == 1_923_088_384

    qwen_dir = s.MODEL_CONFIGS_DIR / "Qwen__Qwen2.5-7B-Instruct"
    qwen_dims = s.parse_llm_dims(s.primary_config(qwen_dir))
    assert qwen_dims["use_sliding_window"] is False
    assert qwen_dims["sliding_window"] == 0


def test_diffusers_aliases_and_video_time_tokens():
    z_image = _summary(s.build_model_payload("Tongyi-MAI__Z-Image", {}))
    assert z_image["Transformer 宽度"] == "3840"
    assert z_image["去噪层数"] == "32"
    assert z_image["latent tokens"] == "4096"

    flux = _summary(s.build_model_payload("black-forest-labs__FLUX.2-klein-9B", {}))
    assert flux["去噪层数"] == "32"
    assert flux["VAE spatial scale"] == "16x16"
    assert flux["latent tokens"] == "4096"

    lingbot_payload = s.build_model_payload("Robbyant__lingbot-video-moe-30b-a3b", {})
    lingbot = _summary(lingbot_payload)
    assert lingbot["去噪层数"] == "48"
    assert lingbot["latent frames"] == "21"
    assert lingbot["latent tokens"] == "86016"
    assert any(control["name"] == "frames" for control in lingbot_payload["controls"])


def test_mla_graph_uses_qk_dimension_and_caches_rope_key():
    payload = s.build_model_payload("ZhipuAI__GLM-5.2", {})
    nodes = {node["id"]: node for node in payload["graph"]["nodes"]}
    assert nodes["mla_q_proj"]["outputShape"].endswith(", 256]")
    assert "K_rope" in nodes["mla_kv_compress"]["outputShape"]


def test_multimodal_merge_rounds_each_grid_axis():
    payload = s.build_model_payload(
        "Qwen__Qwen3-VL-4B-Instruct",
        {"image_height": ["225"], "image_width": ["225"]},
    )
    processor = next(node for node in payload["graph"]["nodes"] if node["id"] == "image_processor")
    details = {item["label"]: item["value"] for item in processor["details"]}
    assert details["raw_patch_count"] == "225"
    assert details["merged_token_count"] == "64"
