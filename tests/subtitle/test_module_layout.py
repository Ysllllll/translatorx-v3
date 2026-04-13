import importlib

import pytest


def test_new_subtitle_modules_exist_and_old_paths_are_removed() -> None:
    assert importlib.import_module("subtitle.model")
    assert importlib.import_module("subtitle.align")
    assert importlib.import_module("subtitle.build")
    assert importlib.import_module("subtitle.io.srt")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle._types")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle.words")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle.builder")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("subtitle.readers.srt")
