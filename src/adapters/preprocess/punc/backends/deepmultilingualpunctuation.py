"""`deepmultilingualpunctuation` backend — multilingual NER fullstop model.

Model weights are cached process-wide per checkpoint name; inference
is serialized per-model via a dedicated :class:`threading.Lock` because
the underlying HuggingFace pipeline is not thread-safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from adapters.preprocess.punc.registry import Backend, PuncBackendRegistry

logger = logging.getLogger(__name__)


LIBRARY_NAME = "deepmultilingualpunctuation"
DEFAULT_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"


_models: dict[str, Any] = {}
_load_lock = threading.Lock()
_infer_locks: dict[str, threading.Lock] = {}


def _load_model(model_name: str) -> Any:
    cached = _models.get(model_name)
    if cached is not None:
        return cached
    with _load_lock:
        cached = _models.get(model_name)
        if cached is not None:
            return cached
        logger.info("Loading NER punctuation model: %s", model_name)
        from deepmultilingualpunctuation import PunctuationModel

        model = PunctuationModel(model=model_name)
        _models[model_name] = model
        _infer_locks[model_name] = threading.Lock()
        return model


@PuncBackendRegistry.register(LIBRARY_NAME)
def factory(*, model: str = DEFAULT_MODEL) -> Backend:
    """Build a ``deepmultilingualpunctuation`` backend.

    Parameters
    ----------
    model:
        HuggingFace model id or local path. Defaults to the multilingual
        fullstop model.
    """
    loaded = _load_model(model)
    lock = _infer_locks[model]

    def _call(texts: list[str]) -> list[str]:
        with lock:
            return [loaded.restore_punctuation(t) for t in texts]

    return _call


__all__ = ["LIBRARY_NAME", "DEFAULT_MODEL", "factory"]
