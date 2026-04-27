"""Tests for /api/pipelines/* and /api/stages/* (Phase 2 / B5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app.app import App
from api.service.app import create_app
from application.config import AppConfig, AuthKeyEntry, ContextEntry, EngineEntry, ServiceConfig, StoreConfig


SAMPLE_PIPELINE = {
    "name": "standard_translate",
    "defaults": {"source_lang": "en", "target_lang": "zh"},
    "build": {"stage": "from_srt", "params": {"path": "{{ input_path }}", "language": "{{ source_lang }}"}},
    "structure": [{"stage": "merge", "params": {"max_len": 80}}],
    "enrich": [{"stage": "translate", "params": {"src": "en", "tgt": "zh"}}],
}


@pytest.fixture
def client(tmp_path):
    pdir = tmp_path / "pipelines"
    pdir.mkdir()
    (pdir / "from_disk.yaml").write_text("name: from_disk\nbuild:\n  stage: from_srt\n  params: {path: /tmp/x.srt, language: en}\n", encoding="utf-8")

    cfg = AppConfig(
        store=StoreConfig(root=str(tmp_path)),
        engines={"default": EngineEntry(model="test-model", base_url="http://localhost", api_key="sk-xxx")},
        contexts={"en-zh": ContextEntry(src="en", tgt="zh")},
        service=ServiceConfig(api_keys={"user-key": AuthKeyEntry(user_id="user-a", tier="free")}),
        pipelines={"inline": SAMPLE_PIPELINE},
        pipelines_dir=str(pdir),
    )
    app = App(cfg)
    api = create_app(app, api_keys={"user-key": ("user-a", "free")})
    with TestClient(api) as c:
        yield c


def _hdr(key: str = "user-key") -> dict:
    return {"X-API-Key": key}


class TestListPipelines:
    def test_lists_inline_and_disk_pipelines(self, client: TestClient):
        r = client.get("/api/pipelines", headers=_hdr())
        assert r.status_code == 200
        body = r.json()
        assert "inline" in body["pipelines"]
        assert "from_disk" in body["pipelines"]


class TestGetPipeline:
    def test_get_inline(self, client: TestClient):
        r = client.get("/api/pipelines/inline", headers=_hdr())
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "inline"
        assert body["definition"]["build"]["stage"] == "from_srt"

    def test_404_unknown(self, client: TestClient):
        r = client.get("/api/pipelines/ghost", headers=_hdr())
        assert r.status_code == 404


class TestValidatePipeline:
    def test_validate_yaml_ok(self, client: TestClient):
        yaml_text = "build:\n  stage: from_srt\n  params: {path: /tmp/x.srt, language: en}\nstructure:\n  - {stage: merge, params: {max_len: 80}}\n"
        r = client.post("/api/pipelines/validate", json={"yaml": yaml_text}, headers=_hdr())
        assert r.status_code == 200
        assert r.json() == {"ok": True, "issues": []}

    def test_validate_yaml_unknown_stage(self, client: TestClient):
        yaml_text = "build:\n  stage: ghost\n  params: {}\n"
        r = client.post("/api/pipelines/validate", json={"yaml": yaml_text}, headers=_hdr())
        body = r.json()
        assert body["ok"] is False
        assert any("ghost" in i["message"] for i in body["issues"])

    def test_validate_dict_body(self, client: TestClient):
        r = client.post("/api/pipelines/validate", json=SAMPLE_PIPELINE, headers=_hdr())
        body = r.json()
        # SAMPLE_PIPELINE has a placeholder {{ input_path }} with no
        # default, so the loader must reject before validation runs.
        assert body["ok"] is False

    def test_validate_yaml_parse_error(self, client: TestClient):
        r = client.post("/api/pipelines/validate", json={"yaml": "not: a: pipeline:"}, headers=_hdr())
        body = r.json()
        assert body["ok"] is False
        assert body["issues"]


class TestStagesRouter:
    def test_list_stages(self, client: TestClient):
        r = client.get("/api/stages", headers=_hdr())
        assert r.status_code == 200
        names = r.json()["stages"]
        for expected in ("from_srt", "merge", "translate"):
            assert expected in names

    def test_pipeline_schema(self, client: TestClient):
        r = client.get("/api/stages/schema", headers=_hdr())
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "object"
        assert "build" in body["properties"]
        # registry-bound: build slot is a oneOf discriminator
        assert "oneOf" in body["properties"]["build"]

    def test_stage_schema_known(self, client: TestClient):
        r = client.get("/api/stages/from_srt/schema", headers=_hdr())
        assert r.status_code == 200
        body = r.json()
        assert "path" in body["properties"]
        assert "path" in body["required"]

    def test_stage_schema_unknown(self, client: TestClient):
        r = client.get("/api/stages/ghost/schema", headers=_hdr())
        assert r.status_code == 404


class TestAuthEnforced:
    def test_pipelines_requires_auth(self, client: TestClient):
        # No X-API-Key header → 401/403
        r = client.get("/api/pipelines")
        assert r.status_code in (401, 403)

    def test_stages_requires_auth(self, client: TestClient):
        r = client.get("/api/stages")
        assert r.status_code in (401, 403)
