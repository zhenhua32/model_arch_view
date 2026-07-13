"""Unit tests for classify_model_dir.

Regression coverage for the classifier logic, in particular the rule that a
model_index.json alone is NOT enough to be treated as a diffusers pipeline --
a model that ships model_index.json without any substantive diffusers component
directory (e.g. a block-diffusion text model such as DiffusionGemma) must fall
through to the LLM/multimodal branches and use its real config.json.
"""
from __future__ import annotations

import pytest

from conftest import all_model_dirs

REAL_DIFFUSERS = [
    "Qwen__Qwen-Image-2512",
    "black-forest-labs__FLUX.2-klein-9B",
    "Tongyi-MAI__Z-Image",
    "Robbyant__lingbot-video-moe-30b-a3b",
    "Wan-AI__Wan2.2-I2V-A14B",
    "amap_cvlab__ABot-World-0-5B-LF",
    "meituan-longcat__LongCat-Video-Avatar-1.5",
]


def test_real_diffusers_models_classified_as_diffusers(serve):
    for mid in REAL_DIFFUSERS:
        assert serve.classify_model_dir(serve.MODEL_CONFIGS_DIR / mid) == "diffusers", mid


def test_model_index_without_components_is_not_diffusers(serve):
    """A model_index.json lacking substantive components must not be 'diffusers'.

    DiffusionGemma ships model_index.json (model/processor/scheduler) but is a
    text model with a real config.json -> it must be classified via config, not as
    a (broken) diffusers graph.
    """
    mid = "google__diffusiongemma-26B-A4B-it"
    assert serve.classify_model_dir(serve.MODEL_CONFIGS_DIR / mid) != "diffusers"


def test_diffusiongemma_builds_from_real_config(serve):
    """DiffusionGemma must parse real dims, not a zero-width diffusers graph."""
    p = serve.build_model_payload("google__diffusiongemma-26B-A4B-it", {})
    summary = {it["label"]: it["value"] for it in p["model"]["summary"]}
    # The language hidden dim is taken from the real config, not a default.
    assert summary.get("语言隐藏维度") not in (None, "?", "0", 0)
    # A diffusers misclassification would leave transformer_width at 0/empty.
    assert "transformer" not in [n.get("id") for n in p["graph"]["nodes"]]


def test_substantive_component_keys_are_required():
    """The classifier's allowed component keys must exclude generic scheduler/tokenizer."""
    from serve_model_arch import _DIFFUSERS_COMPONENT_KEYS

    assert "transformer" in _DIFFUSERS_COMPONENT_KEYS
    assert "vae" in _DIFFUSERS_COMPONENT_KEYS
    # Too generic to be the sole signal of a diffusers pipeline.
    assert "scheduler" not in _DIFFUSERS_COMPONENT_KEYS
    assert "tokenizer" not in _DIFFUSERS_COMPONENT_KEYS


@pytest.mark.parametrize(
    "mid",
    [
        "Comfy-Org__Krea-2",
        "jd-opensource__JoyAI-Image-Edit",
        "Lightricks__LTX-2.3",
    ],
)
def test_metadata_only_or_checkpoint_manifest_is_unknown(serve, mid):
    assert serve.classify_model_dir(serve.MODEL_CONFIGS_DIR / mid) == "unknown"


@pytest.mark.parametrize(
    ("mid", "expected_type", "base_dir"),
    [
        ("jonathanfu__Krea-2-Turbo-zishi", "diffusers", "krea__Krea-2-Turbo"),
        ("laonansheng__ruanqing-Z-Image-Turbo-Tongyi-MAI-v1.0", "diffusers", "Tongyi-MAI__Z-Image-Turbo"),
        ("unsloth__Qwen3.6-27B-MTP-GGUF", "multimodal", "Qwen__Qwen3.6-27B"),
        ("unsloth__Qwen3.6-35B-A3B-GGUF", "multimodal", "Qwen__Qwen3.6-35B-A3B"),
    ],
)
def test_model_card_base_model_inherits_local_architecture(serve, mid, expected_type, base_dir):
    model_dir = serve.MODEL_CONFIGS_DIR / mid
    assert serve.architecture_config_dir(model_dir).name == base_dir
    assert serve.classify_model_dir(model_dir) == expected_type


@pytest.mark.parametrize(
    "model_dir",
    [d for d in all_model_dirs() if (d / "model_index.json").exists()],
    ids=lambda d: d.name,
)
def test_every_model_index_dir_is_consistent(serve, model_dir):
    """A pipeline needs model-index components or a numeric diffusion config."""
    import serve_model_arch as s

    index_data = s.read_json_file(model_dir / "model_index.json") or {}
    is_diffusers = serve.classify_model_dir(model_dir) == "diffusers"
    has_component = any(k in index_data for k in s._DIFFUSERS_COMPONENT_KEYS)
    has_numeric_transformer = bool(s.discover_diffusion_transformer_configs(model_dir))
    assert is_diffusers == (has_component or has_numeric_transformer), (
        f"{model_dir.name}: classifier said diffusers={is_diffusers} but "
        f"index has substantive component={has_component} and "
        f"numeric transformer={has_numeric_transformer}"
    )
