"""Lightweight import / smoke tests for arq task backend.

Full end-to-end arq tests require Redis + arq worker and are intentionally
out of scope for the unit suite. These tests verify the module imports
and config wiring are intact.
"""

from __future__ import annotations

import pytest


def test_arq_module_imports_optional():
    arq = pytest.importorskip("arq")
    from api.service.runtime import tasks_arq

    assert hasattr(tasks_arq, "ArqTaskManager")
    assert hasattr(tasks_arq, "build_worker_settings")
    assert callable(tasks_arq._worker_run_task)
    _ = arq  # silence


def test_service_config_task_backend_default():
    from application.config import ServiceConfig

    cfg = ServiceConfig()
    assert cfg.task_backend == "inproc"
    assert cfg.arq_queue_name == "trx:tasks"


def test_service_config_task_backend_arq_roundtrip():
    from application.config import ServiceConfig

    cfg = ServiceConfig.model_validate({"task_backend": "arq", "redis_url": "redis://localhost:6379/0", "arq_queue_name": "x"})
    assert cfg.task_backend == "arq"
    assert cfg.arq_queue_name == "x"


def test_from_app_config_arq_requires_redis_url(tmp_path):
    from api.app.app import App
    from api.service.app import from_app_config
    from application.config import AppConfig, ServiceConfig, StoreConfig

    cfg = AppConfig(service=ServiceConfig(task_backend="arq"), store=StoreConfig(root=str(tmp_path)))
    app = App(cfg)
    with pytest.raises(ValueError, match="arq.*redis_url"):
        from_app_config(app)
