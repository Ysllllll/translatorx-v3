"""FastAPI service layer — HTTP + SSE endpoints for translatorx."""

from api.service.app import create_app, from_app_config

__all__ = ["create_app", "from_app_config"]
