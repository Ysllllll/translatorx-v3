"""OpenAPI schema assertions for /api/pipelines and /api/stages (Phase 2 / B5).

Verifies that response models + examples actually surface in the
``/openapi.json`` document so editor frontends and code generators see
useful information instead of bare ``dict``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app.app import App
from api.service.app import create_app
from application.config import AppConfig, EngineEntry, StoreConfig


@pytest.fixture
def client(tmp_path):
    cfg = AppConfig(store=StoreConfig(root=str(tmp_path)), engines={"default": EngineEntry(model="m", base_url="http://localhost", api_key="k")})
    app = App(cfg)
    api = create_app(app)
    with TestClient(api) as c:
        yield c


def _spec(client: TestClient) -> dict:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    return r.json()


def _resolve_ref(spec: dict, ref: str) -> dict:
    assert ref.startswith("#/")
    node: dict = spec
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def test_list_pipelines_response_has_typed_schema(client: TestClient):
    spec = _spec(client)
    op = spec["paths"]["/api/pipelines"]["get"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    assert schema["title"] == "PipelineListResponse"
    props = schema["properties"]
    assert "pipelines" in props and "tenant" in props


def test_get_pipeline_response_has_example(client: TestClient):
    spec = _spec(client)
    op = spec["paths"]["/api/pipelines/{name}"]["get"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    assert schema["title"] == "PipelineGetResponse"
    assert "example" in schema
    assert schema["example"]["name"] == "standard_translate"


def test_validate_response_includes_issue_model(client: TestClient):
    spec = _spec(client)
    op = spec["paths"]["/api/pipelines/validate"]["post"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    assert schema["title"] == "ValidateResponse"
    # ValidationIssue should be referenced by the issues array.
    components = spec.get("components", {}).get("schemas", {})
    assert "ValidationIssue" in components


def test_list_pipelines_has_tenant_query_param(client: TestClient):
    spec = _spec(client)
    op = spec["paths"]["/api/pipelines"]["get"]
    params = {p["name"]: p for p in op.get("parameters", [])}
    assert "tenant" in params
    assert params["tenant"]["in"] == "query"
    assert params["tenant"]["required"] is False


def test_list_stages_response_has_typed_schema(client: TestClient):
    spec = _spec(client)
    op = spec["paths"]["/api/stages"]["get"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])
    assert schema["title"] == "StageListResponse"
    assert "stages" in schema["properties"]


def test_pipelines_tag_present(client: TestClient):
    spec = _spec(client)
    assert spec["paths"]["/api/pipelines"]["get"]["tags"] == ["pipelines"]
    assert spec["paths"]["/api/stages"]["get"]["tags"] == ["stages"]
