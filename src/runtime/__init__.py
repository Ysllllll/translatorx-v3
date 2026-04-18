"""TranslatorX v3 runtime package.

Orchestrates Processor chains, persists intermediate state via Store, and
provides streaming/batch/course-level execution.

Public API will expand as Stage 3..6 land. Stage 2 introduces the
Workspace (file routing) and Store (state persistence) layers.
"""

from runtime.base import ProcessorBase
from runtime.errors import (
    EngineError,
    ErrorCategory,
    ErrorInfo,
    ErrorReporter,
    PermanentEngineError,
    TransientEngineError,
)
from runtime.progress import (
    ProgressCallback,
    ProgressEvent,
    ProgressKind,
)
from runtime.protocol import (
    Priority,
    Processor,
    Source,
    VideoKey,
)
from runtime.reporters import (
    ChainReporter,
    JsonlErrorReporter,
    LoggerReporter,
)
from runtime.resource_manager import (
    DEFAULT_TIERS,
    BudgetDecision,
    InMemoryResourceManager,
    ResourceManager,
    UsageSnapshot,
    UserTier,
)
from runtime.store import (
    IncompatibleStoreError,
    JsonFileStore,
    SCHEMA_VERSION,
    Store,
    empty_course_data,
    empty_video_data,
    set_nested,
)
from runtime.usage import (
    CompletionResult,
    Usage,
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
    # base
    "ProcessorBase",
    # errors
    "EngineError",
    "ErrorCategory",
    "ErrorInfo",
    "ErrorReporter",
    "PermanentEngineError",
    "TransientEngineError",
    # progress
    "ProgressCallback",
    "ProgressEvent",
    "ProgressKind",
    # protocol
    "Priority",
    "Processor",
    "Source",
    "VideoKey",
    # reporters
    "ChainReporter",
    "JsonlErrorReporter",
    "LoggerReporter",
    # resource manager
    "BudgetDecision",
    "DEFAULT_TIERS",
    "InMemoryResourceManager",
    "ResourceManager",
    "UsageSnapshot",
    "UserTier",
    # store
    "IncompatibleStoreError",
    "JsonFileStore",
    "SCHEMA_VERSION",
    "Store",
    "empty_course_data",
    "empty_video_data",
    "set_nested",
    # usage
    "CompletionResult",
    "Usage",
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
