"""TranslatorX v3 runtime package.

Orchestrates Processor chains, persists intermediate state via Store, and
provides streaming/batch/course-level execution.

Public API will expand as Stage 3..6 land. Stage 2 introduces the
Workspace (file routing) and Store (state persistence) layers.
"""

from runtime.store import (
    IncompatibleStoreError,
    JsonFileStore,
    SCHEMA_VERSION,
    Store,
    empty_course_data,
    empty_video_data,
    set_nested,
)
from runtime.workspace import (
    SubDir,
    SubDirSpec,
    Workspace,
    canonical_key,
    extract_id,
    register_subdir,
    registered_specs,
    strip_id,
    strip_lang_tail,
)

__all__ = [
    # store
    "IncompatibleStoreError",
    "JsonFileStore",
    "SCHEMA_VERSION",
    "Store",
    "empty_course_data",
    "empty_video_data",
    "set_nested",
    # workspace
    "SubDir",
    "SubDirSpec",
    "Workspace",
    "canonical_key",
    "extract_id",
    "register_subdir",
    "registered_specs",
    "strip_id",
    "strip_lang_tail",
]
