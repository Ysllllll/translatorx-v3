"""Tests for application.scheduler.tenant — Phase 5 L1."""

from __future__ import annotations

import pytest

from application.scheduler import DEFAULT_QUOTAS, DEFAULT_TENANT_ID, TenantContext, TenantQuota


class TestTenantQuota:
    def test_defaults_exist_for_all_three_tiers(self) -> None:
        assert set(DEFAULT_QUOTAS) == {"free", "standard", "premium"}

    def test_quota_is_frozen(self) -> None:
        q = DEFAULT_QUOTAS["free"]
        with pytest.raises(Exception):  # FrozenInstanceError on dataclass
            q.max_concurrent_streams = 99  # type: ignore[misc]

    def test_qos_tiers_strictly_increasing_concurrency(self) -> None:
        free = DEFAULT_QUOTAS["free"].max_concurrent_streams
        std = DEFAULT_QUOTAS["standard"].max_concurrent_streams
        prem = DEFAULT_QUOTAS["premium"].max_concurrent_streams
        assert free < std < prem

    def test_premium_has_cost_budget(self) -> None:
        assert DEFAULT_QUOTAS["premium"].cost_budget_usd_per_min is not None
        assert DEFAULT_QUOTAS["free"].cost_budget_usd_per_min is None

    def test_custom_quota_construction(self) -> None:
        q = TenantQuota(max_concurrent_streams=8, max_qps=12.5, qos_tier="standard", cost_budget_usd_per_min=2.5)
        assert q.max_concurrent_streams == 8
        assert q.qos_tier == "standard"


class TestTenantContext:
    def test_default_anonymous_context(self) -> None:
        ctx = TenantContext()
        assert ctx.tenant_id == DEFAULT_TENANT_ID
        assert ctx.quota.qos_tier == "free"
        assert ctx.labels == {}

    def test_labels_isolated_per_instance(self) -> None:
        a = TenantContext(tenant_id="a")
        b = TenantContext(tenant_id="b")
        a.labels["region"] = "us-west"
        assert "region" not in b.labels

    def test_explicit_quota_attached(self) -> None:
        quota = DEFAULT_QUOTAS["premium"]
        ctx = TenantContext(tenant_id="acme", quota=quota)
        assert ctx.tenant_id == "acme"
        assert ctx.quota is quota
