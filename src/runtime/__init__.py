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
from runtime.processors import SummaryProcessor, TranslateProcessor
from runtime.processors.prefix import (
    EN_ZH_PREFIX_RULES,
    PrefixHandler,
    PrefixRule,
    TranslateNodeConfig,
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
from runtime.orchestrator import StreamingOrchestrator, VideoOrchestrator, VideoResult
from runtime.course import (
    CourseOrchestrator,
    CourseResult,
    ProcessorsFactory,
    VideoSpec,
)
from app import App, CourseBuilder, LiveStreamHandle, StreamBuilder, VideoBuilder
from runtime.config import (
    AppConfig,
    ContextEntry,
    EngineEntry,
    PreprocessConfig,
    RuntimeConfig,
    StoreConfig,
)
from runtime.sources import (
    PushQueueSource,
    SrtSource,
    WhisperXSource,
)
from runtime.store import (
    FINGERPRINT_CHAIN,
    IncompatibleStoreError,
    JsonFileStore,
    SCHEMA_VERSION,
    Store,
    empty_course_data,
    empty_video_data,
    get_stale_steps,
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
    # processors
    "SummaryProcessor",
    "TranslateProcessor",
    "EN_ZH_PREFIX_RULES",
    "PrefixHandler",
    "PrefixRule",
    "TranslateNodeConfig",
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
    # orchestrator
    "StreamingOrchestrator",
    "VideoOrchestrator",
    "VideoResult",
    # course
    "CourseOrchestrator",
    "CourseResult",
    "ProcessorsFactory",
    "VideoSpec",
    # app + builders
    "App",
    "CourseBuilder",
    "LiveStreamHandle",
    "StreamBuilder",
    "VideoBuilder",
    # config
    "AppConfig",
    "ContextEntry",
    "EngineEntry",
    "PreprocessConfig",
    "RuntimeConfig",
    "StoreConfig",
    # sources
    "PushQueueSource",
    "SrtSource",
    "WhisperXSource",
    # store
    "FINGERPRINT_CHAIN",
    "IncompatibleStoreError",
    "JsonFileStore",
    "SCHEMA_VERSION",
    "Store",
    "empty_course_data",
    "empty_video_data",
    "get_stale_steps",
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
