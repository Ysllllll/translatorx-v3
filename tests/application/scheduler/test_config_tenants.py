"""Tests for AppConfig.tenants + build_tenant_quotas — Phase 5 L1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from application.config import AppConfig
from application.scheduler import TenantQuota


YAML_WITH_TENANTS = """
tenants:
  acme:
    max_concurrent_streams: 8
    max_qps: 8.0
    qos_tier: premium
    cost_budget_usd_per_min: 5.0
  free_user:
    max_concurrent_streams: 1
    max_qps: 1.0
    qos_tier: free
"""


class TestTenantsConfig:
    def test_default_tenants_is_empty(self) -> None:
        cfg = AppConfig.from_yaml("")
        assert cfg.tenants == {}
        assert cfg.build_tenant_quotas() == {}

    def test_tenants_parse_from_yaml(self) -> None:
        cfg = AppConfig.from_yaml(YAML_WITH_TENANTS)
        assert set(cfg.tenants) == {"acme", "free_user"}
        assert cfg.tenants["acme"].qos_tier == "premium"
        assert cfg.tenants["acme"].cost_budget_usd_per_min == 5.0

    def test_build_tenant_quotas_returns_frozen_dataclasses(self) -> None:
        cfg = AppConfig.from_yaml(YAML_WITH_TENANTS)
        quotas = cfg.build_tenant_quotas()
        assert isinstance(quotas["acme"], TenantQuota)
        assert quotas["acme"].max_concurrent_streams == 8
        assert quotas["acme"].qos_tier == "premium"
        assert quotas["free_user"].cost_budget_usd_per_min is None

    def test_invalid_qos_tier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.from_yaml("tenants:\n  bad:\n    qos_tier: vip\n")

    def test_zero_max_concurrent_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.from_yaml("tenants:\n  bad:\n    max_concurrent_streams: 0\n")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.from_yaml("tenants:\n  bad:\n    max_concurrent_streams: 1\n    bogus: 1\n")

    def test_env_override_for_tenants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Existing tenant with one knob bumped via env.
        monkeypatch.setenv("TRX_TENANTS__ACME__MAX_CONCURRENT_STREAMS", "32")
        cfg = AppConfig.from_yaml("tenants:\n  acme:\n    max_concurrent_streams: 4\n    qos_tier: premium\n")
        assert cfg.tenants["acme"].max_concurrent_streams == 32
        assert cfg.tenants["acme"].qos_tier == "premium"
