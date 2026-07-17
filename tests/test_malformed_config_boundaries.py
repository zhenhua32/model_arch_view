from __future__ import annotations

import json
from pathlib import Path

import serve_model_arch as s


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_simple_yaml_keeps_tagged_and_plain_mappings_separate(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: !new:example.Model\n"
        "  hidden_size: 128\n"
        "encoder:\n"
        "  layers: 4\n",
        encoding="utf-8",
    )

    config = s.read_simple_yaml_file(path)

    assert config["model"] == {"_tag": "!new:example.Model", "hidden_size": 128}
    assert config["encoder"] == {"layers": 4}


def test_llm_dimension_parser_rejects_wrong_nested_container_types() -> None:
    config = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 256,
        "vocab_size": 1024,
        "dsa_layers": "0,1",
        "swa_layers": {"0": True},
        "compress_ratios": 4,
    }

    dims = s.parse_llm_dims(config, {"moe": "invalid"})

    assert dims["num_layers"] == 2
    assert dims["num_experts"] == 0
    assert dims["sparse_attention_layers"] == 0
    assert dims["compress_ratios"] == []


def test_asr_builder_ignores_scalar_yaml_encoder_config(tmp_path: Path) -> None:
    write_json(
        tmp_path / "config.json",
        {
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
            "encoder": "invalid",
            "thinker_config": None,
            "head": "invalid",
        },
    )
    (tmp_path / "config.yaml").write_text("audio_encoder_conf: invalid\n", encoding="utf-8")

    payload = s.build_asr_payload(tmp_path, "synthetic-asr", {})

    assert payload["selectedNodeId"] == "audio_encoder"
    assert payload["graph"]["nodes"]


def test_sam_builder_ignores_scalar_nested_configs(tmp_path: Path) -> None:
    write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Sam2VideoModel"],
            "detector_config": {"vision_config": "invalid", "text_config": 7},
            "tracker_config": {"vision_config": [1, 2]},
        },
    )
    write_json(
        tmp_path / "processor_config.json",
        {"image_processor": "invalid", "video_processor": 8},
    )

    payload = s.build_sam_video_payload(tmp_path, "synthetic-sam", {})

    assert payload["selectedNodeId"] == "detector_tracker"
    assert payload["parameters"]["frames"] == 8


def test_multimodal_builder_falls_back_after_malformed_nested_configs(tmp_path: Path) -> None:
    write_json(
        tmp_path / "config.json",
        {
            "architectures": ["VisionLanguageForConditionalGeneration"],
            "model_type": "synthetic_vlm",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 1024,
            "max_position_embeddings": 512,
            "text_config": "invalid",
            "language_config": 42,
            "vision_config": "invalid",
            "projector_config": [128],
            "img_token_compression_config": "invalid",
        },
    )
    write_json(
        tmp_path / "processor_config.json",
        {
            "image_processor": "invalid",
            "video_processor": "invalid",
            "feature_extractor": "invalid",
        },
    )

    payload = s.build_multimodal_payload(tmp_path, "synthetic-vlm", {})

    assert payload["selectedNodeId"] == "fusion_context"
    assert payload["parameters"]["seq_len"] == 512


def test_diffusers_text_helpers_ignore_malformed_nested_config() -> None:
    text_config = {
        "text_config": "invalid",
        "hidden_size": 768,
        "num_hidden_layers": 12,
    }

    assert s.infer_text_hidden(text_config, {}) == 768
    assert s.infer_text_layers(text_config, {}) == 12
