from __future__ import annotations

from pathlib import Path

import pytest

import download_hot_models as downloader


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
        (" owner/model ", "owner/model"),
        ("owner\\model", "owner/model"),
        ("org/subgroup/model", "org/subgroup/model"),
    ],
)
def test_normalize_model_id(raw, expected):
    assert downloader.normalize_model_id(raw) == expected


@pytest.mark.parametrize(
    "model_id",
    ["", "single-name", "/owner/model", "owner//model", "../model", "owner/..", "C:\\temp\\model", "owner/bad:name"],
)
def test_normalize_model_id_rejects_unsafe_paths(model_id):
    with pytest.raises(ValueError):
        downloader.normalize_model_id(model_id)


def test_safe_model_dir_name_has_no_path_separators():
    safe_name = downloader.safe_model_dir_name("org/subgroup/model")
    assert safe_name == "org__subgroup__model"
    assert "/" not in safe_name and "\\" not in safe_name


def test_model_id_from_record_rejects_outer_path_separators():
    with pytest.raises(ValueError):
        downloader.model_id_from_record({"Path": "", "Name": "/owner/model"})


def test_search_models_sends_bounded_modelscope_query(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Success": True,
                "Data": {
                    "Model": {
                        "Models": [
                            {"Path": "owner", "Name": f"model-{index}"}
                            for index in range(25)
                        ]
                    }
                },
            }

    class FakeSession:
        def put(self, url, json, timeout):
            calls.update(url=url, payload=json, timeout=timeout)
            return FakeResponse()

    monkeypatch.setattr(downloader, "create_modelscope_session", lambda token: (calls.update(token=token) or FakeSession()))

    models = downloader.search_models(" Qwen3 ", limit=99, token="secret", timeout=7)

    assert len(models) == 20
    assert calls == {
        "token": "secret",
        "url": downloader.MODELSCOPE_API,
        "payload": {
            "Name": "Qwen3",
            "Criterion": [],
            "SingleCriterion": [],
            "SortBy": "Default",
            "PageNumber": 1,
            "PageSize": 20,
        },
        "timeout": 7,
    }


def test_search_models_skips_short_queries():
    assert downloader.search_models("q") == []


def test_download_model_config_uses_lazy_downloader_and_tracks_changes(tmp_path, monkeypatch):
    local_dir = tmp_path / "model"

    def fake_snapshot_download(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    monkeypatch.setattr(downloader, "get_snapshot_download", lambda: fake_snapshot_download)
    result = downloader.download_model_config(
        " owner/model ",
        str(local_dir),
        ["*.json"],
    )

    assert result["status"] == "success"
    assert result["model_id"] == "owner/model"
    assert result["files"] == [{"file": "config.json", "size": 2}]
    assert result["downloaded_files"] == result["files"]


def test_download_model_config_fails_when_no_directory_is_created(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "get_snapshot_download", lambda: lambda **_kwargs: None)
    result = downloader.download_model_config(
        "owner/model",
        str(tmp_path / "missing"),
        ["*.json"],
    )
    assert result["status"] == "failed"
    assert "有效的模型目录" in result["error"]
