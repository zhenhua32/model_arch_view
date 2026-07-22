from __future__ import annotations

from pathlib import Path

import pytest

import serve_model_arch as serve


def use_empty_catalog(tmp_path, monkeypatch):
    catalog = tmp_path / "model_configs"
    catalog.mkdir()
    monkeypatch.setattr(serve, "MODEL_CONFIGS_DIR", catalog)
    serve.clear_model_config_caches()
    return catalog


def test_modelscope_search_excludes_local_and_invalid_records(tmp_path, monkeypatch):
    catalog = use_empty_catalog(tmp_path, monkeypatch)
    (catalog / "owner__local").mkdir()
    records = (
        {"Path": "owner", "Name": "local", "Downloads": 100},
        {"Path": "owner", "Name": "remote", "ChineseName": "远端模型", "Downloads": "42", "Stars": 3},
        {"Path": "owner", "Name": "remote", "Downloads": 1},
        {"Path": "", "Name": "invalid"},
    )
    monkeypatch.setattr(serve, "cached_modelscope_search", lambda query, limit: records)

    models = serve.search_modelscope_catalog("remote", limit=50)

    assert models == [
        {
            "modelId": "owner/remote",
            "localId": "owner__remote",
            "name": "owner/remote",
            "chineseName": "远端模型",
            "downloads": 42,
            "stars": 3,
            "source": "modelscope",
        }
    ]


def test_modelscope_download_moves_config_snapshot_into_catalog(tmp_path, monkeypatch):
    catalog = use_empty_catalog(tmp_path, monkeypatch)
    captured = {}

    def fake_download_model_config(model_id, local_dir, allow_patterns, token=None):
        target = Path(local_dir)
        target.mkdir(parents=True)
        (target / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
        captured.update(model_id=model_id, allow_patterns=allow_patterns, token=token)
        return {
            "status": "success",
            "path": str(target),
            "files": [{"file": "config.json", "size": 22}],
            "total_size": 22,
            "elapsed": 0.2,
        }

    monkeypatch.setattr(serve.modelscope_downloader, "download_model_config", fake_download_model_config)

    result = serve.download_modelscope_config_to_catalog("owner/model")

    assert result["status"] == "success"
    assert result["model"]["id"] == "owner__model"
    assert result["download"]["fileCount"] == 1
    assert (catalog / "owner__model" / "config.json").is_file()
    assert not list(catalog.glob(".modelscope-*"))
    assert captured["model_id"] == "owner/model"
    assert "*.json" in captured["allow_patterns"]
    assert "*.py" not in captured["allow_patterns"]


def test_modelscope_download_rejects_snapshot_outside_staging(tmp_path, monkeypatch):
    catalog = use_empty_catalog(tmp_path, monkeypatch)
    external = tmp_path / "external"
    external.mkdir()
    (external / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        serve.modelscope_downloader,
        "download_model_config",
        lambda **_kwargs: {
            "status": "success",
            "path": str(external),
            "files": [{"file": "config.json", "size": 2}],
        },
    )

    with pytest.raises(RuntimeError, match="非预期目录"):
        serve.download_modelscope_config_to_catalog("owner/model")

    assert not (catalog / "owner__model").exists()
    assert not list(catalog.glob(".modelscope-*"))
