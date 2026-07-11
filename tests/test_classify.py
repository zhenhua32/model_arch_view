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
    "model_dir",
    [d for d in all_model_dirs() if (d / "model_index.json").exists()],
    ids=lambda d: d.name,
)
def test_every_model_index_dir_is_consistent(serve, model_dir):
    """Each model_index.json must contain a substantive component OR be non-diffusers."""
    import serve_model_arch as s

    index_data = s.read_json_file(model_dir / "model_index.json") or {}
    is_diffusers = serve.classify_model_dir(model_dir) == "diffusers"
    has_component = any(k in index_data for k in s._DIFFUSERS_COMPONENT_KEYS)
    assert is_diffusers == has_component, (
        f"{model_dir.name}: classifier said diffusers={is_diffusers} but "
        f"index has substantive component={has_component}"
    )
