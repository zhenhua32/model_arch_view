from __future__ import annotations

import json

import pytest

import serve_model_arch as s


def test_llm_audit_tracks_formula_sources_and_checkpoint_truth():
    payload = s.build_model_payload("Qwen__Qwen2.5-7B-Instruct", {})
    audit = payload["audit"]
    fields = {item["name"]: item for item in audit["config"]["fields"]}
    evidence = {item["id"]: item for item in audit["evidence"]}

    assert audit["confidence"]["level"] == "high"
    assert audit["checkpoint"]["status"] == "matched"
    assert audit["checkpoint"]["comparable"] is True
    assert audit["checkpoint"]["relativeError"] < 0.001
    assert fields["hidden_size"]["source"] == "config.json:hidden_size"
    assert fields["hidden_size"]["origin"] == "direct"
    assert fields["head_dim"]["origin"] == "derived"
    assert evidence["total-parameters"]["result"] == payload["metrics"]["total_params"]
    assert evidence["memory-footprint"]["result"] == payload["metrics"]["memory"]["total_bytes"]


def test_checkpoint_audit_distinguishes_close_and_quantized_storage():
    pangu = s.build_model_payload("openpangu__openPangu-2.0-Flash", {})["audit"]["checkpoint"]
    quantized = s.build_model_payload("meituan-longcat__LongCat-2.0-FP8", {})["audit"]["checkpoint"]

    assert pangu["status"] == "close"
    assert pangu["comparable"] is True
    assert 0.02 < pangu["relativeError"] < 0.03
    assert quantized["available"] is True
    assert quantized["status"] == "informational"
    assert quantized["comparable"] is False


def test_inherited_config_and_unknown_parser_are_reported():
    inherited = s.build_model_payload("Qwen__Qwen3.6-27B-FP8", {})["audit"]
    unknown = s.build_model_payload("Comfy-Org__Krea-2", {})["audit"]

    assert [item["role"] for item in inherited["config"]["lineage"]] == ["local", "base"]
    assert any(item["code"] == "CONFIG_INHERITED" for item in inherited["diagnostics"])
    assert unknown["confidence"]["level"] == "low"
    assert any(item["code"] == "PARSER_UNSUPPORTED" for item in unknown["diagnostics"])


def test_non_llm_graph_formulas_become_audit_evidence():
    audit = s.build_model_payload("Tongyi-MAI__Z-Image", {})["audit"]
    assert audit["evidence"]
    assert all(item["category"] == "shape" for item in audit["evidence"])
    assert any("latent" in item["formula"].lower() for item in audit["evidence"])


def test_config_alias_conflict_creates_structured_diagnostic(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2}),
        encoding="utf-8",
    )
    (tmp_path / "params.json").write_text(json.dumps({"dim": 256}), encoding="utf-8")
    payload = {
        "model": {"type": "llm", "sources": ["config.json", "params.json"]},
        "metrics": {
            "total_params": 1,
            "active_params": 1,
            "breakdown": {"attention": 1},
            "formula_terms": {"hidden_size": 128, "num_layers": 2, "num_heads": 2},
        },
        "graph": {"nodes": []},
        "warnings": [],
    }

    audit = s.build_model_audit(tmp_path, payload)
    conflict = next(item for item in audit["diagnostics"] if item["code"] == "CONFIG_ALIAS_CONFLICT")
    assert conflict["severity"] == "warning"
    assert "config.json:hidden_size" in conflict["message"]
    assert "params.json:dim" in conflict["message"]


@pytest.mark.parametrize("model_id", ["BAAI__bge-m3", "facebook__sam3.1"])
def test_audit_payload_is_strict_json(model_id):
    payload = s.build_model_payload(model_id, {})
    json.dumps(payload["audit"], allow_nan=False)
