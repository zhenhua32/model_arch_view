"""Every model directory must build a valid payload without raising.

This is the broad smoke test: it catches parsing crashes (e.g. list-valued
patch_size on video diffusers) and structural regressions across all model
families -- llm, multimodal, diffusers and unknown/raw-config fallbacks.
"""
from __future__ import annotations

import json

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
    audit = payload.get("audit")
    assert isinstance(audit, dict), f"{model_id}: missing audit"
    assert audit.get("schemaVersion") == 1
    assert 0 <= audit.get("confidence", {}).get("score", -1) <= 100
    assert audit.get("confidence", {}).get("level") in {"high", "medium", "low"}
    assert isinstance(audit.get("diagnostics"), list)
    assert isinstance(audit.get("evidence"), list)
    assert isinstance(audit.get("config"), dict)

    graph = payload["graph"]
    lanes = graph.get("lanes") or []
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    assert nodes, f"{model_id}: graph has no nodes"
    assert any(not node.get("viewModes") or "summary" in node["viewModes"] for node in nodes), (
        f"{model_id}: graph has no summary-visible nodes"
    )
    lane_ids = [lane["id"] for lane in lanes]
    node_ids = [node["id"] for node in nodes]
    assert len(lane_ids) == len(set(lane_ids)), f"{model_id}: duplicate lane ids"
    assert len(node_ids) == len(set(node_ids)), f"{model_id}: duplicate node ids"
    assert all(node["lane"] in lane_ids for node in nodes), f"{model_id}: node references missing lane"
    assert all(not node.get("parentId") or node["parentId"] in node_ids for node in nodes), (
        f"{model_id}: node references missing parent"
    )
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in edges), (
        f"{model_id}: edge references missing node"
    )
    assert payload.get("selectedNodeId") in node_ids, f"{model_id}: selected node is missing"
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=MODEL_IDS)
def test_summary_items_are_wellformed(serve, model_dir):
    payload = serve.build_model_payload(model_dir.name, {})
    for item in payload["model"]["summary"]:
        assert set(item) >= {"label", "value"}, f"{model_dir.name}: malformed summary item {item!r}"
        assert isinstance(item["label"], str) and item["label"]
        assert isinstance(item["value"], str)


@pytest.mark.parametrize("model_dir", MODEL_DIRS, ids=MODEL_IDS)
def test_control_values_obey_declared_domains(serve, model_dir):
    model_id = model_dir.name
    controls = serve.build_model_payload(model_id, {})["controls"]
    low_query = {}
    high_query = {}
    for control in controls:
        if control.get("type") == "select":
            low_query[control["name"]] = ["__invalid_option__"]
            high_query[control["name"]] = ["__invalid_option__"]
        else:
            low_query[control["name"]] = ["-999999999"]
            high_query[control["name"]] = ["999999999"]

    for payload in (
        serve.build_model_payload(model_id, low_query),
        serve.build_model_payload(model_id, high_query),
    ):
        for control in payload["controls"]:
            value = control.get("value")
            if control.get("type") == "select":
                assert value in control.get("options", []), f"{model_id}: invalid {control['name']} option"
                continue
            if control.get("min") is not None:
                assert value >= control["min"], f"{model_id}: {control['name']} below minimum"
            if control.get("max") is not None:
                assert value <= control["max"], f"{model_id}: {control['name']} above maximum"
