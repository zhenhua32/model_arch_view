"""Regression checks tied to real configs that exposed estimator bugs."""
from __future__ import annotations

import serve_model_arch as s


def _summary(payload):
    return {item["label"]: item["value"] for item in payload["model"]["summary"]}


def test_hy3_shared_expert_matches_model_card_scale():
    payload = s.build_model_payload("Tencent-Hunyuan__Hy3", {})
    metrics = payload["metrics"]
    assert metrics["total_params"] == 298_688_970_752
    assert metrics["active_params"] == 20_117_979_136
    assert metrics["breakdown"]["shared_experts"] > 0
    assert 298 <= metrics["total_params"] / 1e9 <= 299
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


def test_qwen_embedding_keeps_causal_decoder_but_removes_generation_head():
    payload = s.build_model_payload("Qwen__Qwen3-Embedding-8B", {})
    metrics = payload["metrics"]
    node_labels = {node["label"] for node in payload["graph"]["nodes"]}
    checkpoint_bytes = s.read_json_file(
        s.MODEL_CONFIGS_DIR / "Qwen__Qwen3-Embedding-8B" / "model.safetensors.index.json"
    )["metadata"]["total_size"]

    assert payload["model"]["headline"].startswith("Decoder-only 表征模型")
    assert metrics["causal_attention"] is True
    assert metrics["output_head_params"] == 0
    assert metrics["kv_cache_mb_per_1k"] is None
    assert metrics["throughput"] == []
    assert abs(metrics["total_params"] * 2 - checkpoint_bytes) / checkpoint_bytes < 0.001
    assert "Embedding Pooling" in node_labels
    assert "LM Head" not in node_labels


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


def test_wan_configs_without_model_index_use_video_latent_math():
    abot = s.build_model_payload("amap_cvlab__ABot-World-0-5B-LF", {})
    abot_summary = _summary(abot)
    assert abot_summary["Transformer 宽度"] == "3072"
    assert abot_summary["去噪层数"] == "30"
    assert abot_summary["latent frames"] == "21"
    assert abot_summary["latent tokens"] == "86016"
    text_condition = next(node for node in abot["graph"]["nodes"] if node["id"] == "text_condition")
    assert text_condition["outputShape"] == "[1, 256, 4096]"
    assert "image_condition_input" in {node["id"] for node in abot["graph"]["nodes"]}

    i2v = s.build_model_payload("Wan-AI__Wan2.2-I2V-A14B", {})
    assert _summary(i2v)["去噪阶段"] == "2"
    assert _summary(i2v)["latent tokens"] == "86016"

    t2v = s.build_model_payload("Wan-AI__Wan2.2-T2V-A14B", {})
    assert "image_condition_input" not in {node["id"] for node in t2v["graph"]["nodes"]}


def test_longcat_avatar_uses_audio_conditioning_dimensions():
    payload = s.build_model_payload("meituan-longcat__LongCat-Video-Avatar-1.5", {})
    summary = _summary(payload)
    assert summary["Transformer 宽度"] == "4096"
    assert summary["去噪层数"] == "48"
    assert summary["音频条件"] == "32 x 1280"
    nodes = {node["id"]: node for node in payload["graph"]["nodes"]}
    assert nodes["audio_condition_input"]["outputShape"] == "[1, 32, 1280]"
    assert payload["parameters"]["audio_tokens"] == 32


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


def test_multimodal_pixel_bounds_are_not_image_edges_and_modality_is_selected():
    image = s.build_model_payload("InternScience__Agents-A1", {})
    assert image["parameters"]["image_height"] == 1024
    assert image["parameters"]["image_width"] == 1024
    assert image["parameters"]["modality"] == "image"
    assert _summary(image)["当前总 tokens"] == "2048"

    video = s.build_model_payload(
        "InternScience__Agents-A1",
        {"modality": ["video"]},
    )
    assert _summary(video)["当前总 tokens"] == "5120"


def test_fixed_visual_tokens_and_nested_preprocessor_aliases():
    diffusion_gemma = s.build_model_payload("google__diffusiongemma-26B-A4B-it", {})
    assert _summary(diffusion_gemma)["当前总 tokens"] == "1304"
    audio = s.build_model_payload(
        "google__diffusiongemma-26B-A4B-it",
        {"modality": ["audio"]},
    )
    assert audio["parameters"]["audio_tokens"] == 750
    assert _summary(audio)["当前总 tokens"] == "1774"

    minimax = s.build_model_payload("MiniMax__MiniMax-M3", {})
    assert minimax["parameters"]["image_height"] == 672
    assert _summary(minimax)["当前总 tokens"] == "1600"

    mistral = s.build_model_payload("mistralai__Mistral-Medium-3.5-128B", {})
    assert _summary(mistral)["当前总 tokens"] == "4049"
    kimi = s.build_model_payload("moonshotai__Kimi-K2.6", {})
    assert _summary(kimi)["当前总 tokens"] == "2393"
    step = s.build_model_payload("stepfun-ai__Step-3.7-Flash", {})
    assert _summary(step)["当前总 tokens"] == "1193"


def test_audio_segmentation_and_tts_models_use_dedicated_payloads():
    cohere = s.build_model_payload("CohereLabs__cohere-transcribe-03-2026", {})
    assert _summary(cohere)["类型"] == "语音识别"
    assert cohere["parameters"]["feature_frames"] == 3000
    assert cohere["parameters"]["audio_tokens"] == 375
    assert "image_input" not in {node["id"] for node in cohere["graph"]["nodes"]}
    asr_logits = next(node for node in cohere["graph"]["nodes"] if node["id"] == "asr_logits")
    assert asr_logits["outputShape"].endswith(", 16384]")

    qwen = s.build_model_payload("Qwen__Qwen3-ASR-1.7B", {})
    assert qwen["model"]["type"] == "multimodal"
    assert qwen["parameters"]["audio_tokens"] == 1500

    fun_asr = s.build_model_payload("FunAudioLLM__Fun-ASR-Nano-2512", {})
    assert fun_asr["model"]["architecture"] == "FunASRNano"
    assert fun_asr["parameters"]["audio_tokens"] == 500

    sam = s.build_model_payload("facebook__sam3.1", {})
    assert _summary(sam)["类型"] == "视频分割"
    assert sam["parameters"]["patch_size"] == 14
    assert sam["parameters"]["tokens_per_frame"] == 5184
    assert sam["parameters"]["prompt_tokens"] == 16
    assert "lm_head" not in {node["id"] for node in sam["graph"]["nodes"]}

    moss = s.build_model_payload("openmoss__MOSS-TTS", {})
    assert _summary(moss)["类型"] == "语音合成"
    assert moss["parameters"]["n_vq"] == 32
    assert moss["parameters"]["delayed_steps"] == 781


def test_non_delay_tts_families_use_their_real_acoustic_backends():
    cosy = s.build_model_payload("iic__CosyVoice2-0.5B", {})
    cosy_summary = _summary(cosy)
    assert cosy_summary["语言隐藏维度"] == "896"
    assert cosy_summary["主干层数"] == "24"
    assert cosy_summary["声学预测头"] == "1"
    assert cosy_summary["声学后端"] == "Causal CFM + HiFT"
    assert cosy["parameters"]["waveform_samples"] == 720_000
    cosy_nodes = {node["id"]: node for node in cosy["graph"]["nodes"]}
    assert "delay_pattern" not in cosy_nodes
    assert cosy_nodes["flow_decoder"]["outputShape"] == "[1, 1500, 80]"
    assert cosy_nodes["vocoder"]["outputShape"] == "[1, 720000]"

    index = s.build_model_payload("IndexTeam__IndexTTS-2", {})
    index_summary = _summary(index)
    assert index_summary["语言隐藏维度"] == "1280"
    assert index_summary["声学后端"] == "S2Mel DiT + BigVGAN"
    assert index["parameters"]["sample_rate"] == 22_050
    assert "s2mel_dit" in {node["id"] for node in index["graph"]["nodes"]}

    vox = s.build_model_payload("OpenBMB__VoxCPM2", {})
    vox_summary = _summary(vox)
    assert vox_summary["语言隐藏维度"] == "2048"
    assert vox_summary["声学后端"] == "Residual LM + DiT + Audio VAE"
    assert vox["parameters"]["sample_rate"] == 48_000
    assert {"residual_lm", "audio_dit", "audio_vae_decoder"}.issubset(
        {node["id"] for node in vox["graph"]["nodes"]}
    )


def test_tagged_yaml_mappings_keep_their_children():
    config = s.supplemental_yaml_config(s.MODEL_CONFIGS_DIR / "iic__CosyVoice2-0.5B")
    assert config["mel_spec_transform1"]["hop_size"] == 480
    assert config["filter"]["token_max_length"] == 200


def test_hy_world_moe_lists_and_composite_geometry_math():
    model_dir = s.MODEL_CONFIGS_DIR / "Tencent-Hunyuan__HY-World-2.0"
    pano_config = s.read_json_file(model_dir / "HY-Pano-2.0" / "config.json")
    dims = s.parse_llm_dims(pano_config)
    metrics = s.estimate_llm_metrics(dims)
    assert dims["experts_per_tok"] == 8
    assert dims["moe_ffn_hidden"] == 3072
    assert dims["n_shared_experts"] == 1
    assert metrics["total_params"] == 80_950_067_200
    assert metrics["active_params"] == 12_759_072_768

    payload = s.build_model_payload("Tencent-Hunyuan__HY-World-2.0", {})
    summary = _summary(payload)
    assert payload["model"]["architecture"] == "HYWorld2Pipeline"
    assert summary["HY-Pano 主干参数"] == "80.95B / active 12.76B"
    assert payload["parameters"]["pano_latent_tokens"] == 7320
    assert payload["parameters"]["pano_output_width"] == 1920
    assert payload["parameters"]["mirror_total_tokens"] == 10_984
    assert payload["parameters"]["gaussian_count"] == 2_146_592

    custom = s.build_model_payload(
        "Tencent-Hunyuan__HY-World-2.0",
        {"task": ["reconstruction"], "views": ["2"], "recon_height": ["952"], "recon_width": ["714"]},
    )
    assert custom["parameters"]["tokens_per_view"] == 3472
    assert custom["parameters"]["mirror_total_tokens"] == 6944
    assert custom["parameters"]["gaussian_count"] == 1_359_456


def test_nemotron_streaming_chunk_and_cache_math():
    payload = s.build_model_payload(
        "nv-community__nemotron-3.5-asr-streaming-0.6b",
        {"chunk_ms": ["320"], "chunks": ["10"], "language_mode": ["auto"]},
    )
    summary = _summary(payload)
    assert payload["model"]["architecture"] == "FastConformerCacheAwareRNNT"
    assert summary["编码器"] == "24 层 FastConformer / D=512"
    assert payload["parameters"]["chunk_frames"] == 4
    assert payload["parameters"]["right_context_frames"] == 3
    assert payload["parameters"]["left_cache_ms"] == 4480
    assert payload["parameters"]["total_audio_ms"] == 3200
    nodes = {node["id"]: node for node in payload["graph"]["nodes"]}
    assert nodes["prompt_fusion"]["inputShape"] == "[1, 4, 640]"


def test_inherited_lora_and_gguf_payloads_use_base_shapes():
    krea = s.build_model_payload("jonathanfu__Krea-2-Turbo-zishi", {})
    assert _summary(krea)["Transformer 宽度"] == "6144"
    assert any("base_model=krea/Krea-2-Turbo" in warning for warning in krea["warnings"])

    gguf = s.build_model_payload("unsloth__Qwen3.6-35B-A3B-GGUF", {})
    assert gguf["model"]["type"] == "multimodal"
    assert _summary(gguf)["语言隐藏维度"] == "2048"
    assert any("base_model=Qwen/Qwen3.6-35B-A3B" in warning for warning in gguf["warnings"])


def test_inherited_fp8_config_keeps_base_dimensions_and_local_quantization():
    model_dir = s.MODEL_CONFIGS_DIR / "Qwen__Qwen3.6-27B-FP8"
    base_dir = s.local_base_model_dir(model_dir)
    config = s.primary_config(model_dir)
    assert base_dir is not None
    assert config["text_config"]["hidden_size"] == s.primary_config(base_dir)["text_config"]["hidden_size"]
    assert config["quantization_config"]["quant_method"] == "fp8"


def test_deepseek_v4_uses_compressed_mqa_and_mixed_precision_profile():
    flash = s.build_model_payload(
        "deepseek-ai__DeepSeek-V4-Flash",
        {"seq_len": ["1048576"]},
    )
    metrics = flash["metrics"]
    assert 290e9 <= metrics["total_params"] <= 291e9
    assert 13e9 <= metrics["active_params"] <= 13.5e9
    assert metrics["is_deepseek_v4"] is True
    assert metrics["compress_ratios"].count(4) == 21
    assert metrics["compress_ratios"].count(128) == 20
    assert 6 * 1024**3 <= metrics["memory"]["kv_bytes"] <= 7 * 1024**3
    assert metrics["memory"]["weight_format"] == "混合 FP4 experts + FP8 core"
    assert 130 <= metrics["gflops_per_token"] <= 150
    h100 = next(row for row in metrics["throughput"] if row["name"] == "H100 80G")
    assert h100["compute_precision"] == "fp8"
    assert 0.75 <= h100["active_bytes_per_param"] <= 0.76
    bf16 = s.build_model_payload(
        "deepseek-ai__DeepSeek-V4-Flash",
        {"precision": ["bf16"]},
    )["metrics"]["memory"]
    assert bf16["weight_source"] == "parameter_estimate"
    assert bf16["weight_format"] == "BF16"

    pro = s.build_model_payload("deepseek-ai__DeepSeek-V4-Pro", {})["metrics"]
    assert 1.59e12 <= pro["total_params"] <= 1.61e12
    assert 48e9 <= pro["active_params"] <= 50e9


def test_openpangu_sparse_attention_and_minimax_mtp_are_accounted():
    pangu = s.build_model_payload(
        "openpangu__openPangu-2.0-Flash",
        {"seq_len": ["524288"]},
    )["metrics"]
    assert pangu["sparse_attention_layers"] == 16
    assert pangu["sliding_attention_layers"] == 30
    assert pangu["sparse_topk"] == 2048
    assert 8 * 1024**3 <= pangu["memory"]["kv_bytes"] <= 10 * 1024**3
    assert 60 <= pangu["gflops_per_token"] <= 70
    assert pangu["mtp_included"] == pangu["mtp_params"] > 0

    minimax = s.build_model_payload("MiniMax__MiniMax-M2.7", {})["metrics"]
    assert minimax["mtp_layers"] == 3
    assert minimax["mtp_included"] == minimax["mtp_params"]
    assert 239e9 <= minimax["total_params"] <= 241e9
    assert 1.99 <= minimax["memory"]["checkpoint_bytes_per_param"] <= 2.02
    assert minimax["memory"]["weight_format"] == "checkpoint≈BF16 / runtime FP8"
