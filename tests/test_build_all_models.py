"""Every model directory must build a valid payload without raising.

This is the broad smoke test: it catches parsing crashes (e.g. list-valued
patch_size on video diffusers) and structural regressions across all model
families -- llm, multimodal, diffusers and unknown/raw-config fallbacks.
"""
from __future__ import annotations

import pytest

from conftest import all_model_dirs, model_ids

MODEL_DIRS = all_model_dirs()
MODEL_IDS = model_ids()

VALID_TYPES = {"llm", "multimodal", "diffusers", "unknown"}


def test_model_corpus_is_present():
    assert MODEL_DIRS, "no model directories found under model_configs/"


@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=MODEL_IDS)
def test_model_builds_payload(serve, model_dir):
    model_id = model_dir.name

    # Classification must be one of the known buckets.
    model_type = serve.classify_model_dir(model_dir)
    assert model_type in VALID_TYPES, f"{model_id}: unexpected type {model_type!r}"

    # Building must not raise.
    payload = serve.build_model_payload(model_id, {})

    # Structural invariants shared by every payload.
    assert isinstance(payload, dict)
    model = payload.get("model")
    assert isinstance(model, dict), f"{model_id}: missing 'model' block"
    assert model.get("id") == model_id
    assert model.get("type") in VALID_TYPES
    assert isinstance(model.get("summary"), list) and model["summary"], f"{model_id}: empty summary"
    assert isinstance(payload.get("graph"), dict), f"{model_id}: missing graph"
    assert isinstance(payload.get("warnings"), list)


@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=MODEL_IDS)
def test_summary_items_are_wellformed(serve, model_dir):
    payload = serve.build_model_payload(model_dir.name, {})
    for item in payload["model"]["summary"]:
        assert set(item) >= {"label", "value"}, f"{model_dir.name}: malformed summary item {item!r}"
        assert isinstance(item["label"], str) and item["label"]
        assert isinstance(item["value"], str)
