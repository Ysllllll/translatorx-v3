"""Workspace layer — file layout + routing for a single course.

Two-layer design:

* **Workspace** (this module) owns file paths. It knows *where* each file
  type lives on disk, how to find an existing file by key, and how to
  synthesize a destination path for a new file.
* **Store** (``runtime.store``) owns JSON-structured state (per-video
  translation JSON, course-level metadata.json). Store delegates path
  resolution to Workspace.

Everything else (downloaders, transcribers, translators, TTS) reads from
and writes to disk via the paths Workspace provides.

Design refs: D-061 Workspace layer, D-062 registry, D-063 eager course
index, D-064 write-through + stat fallback, D-065 three-level match.

Layout::

    <root>/<course>/                       # course dir
        <video>.mp4                        # home: videos sit at root
        zzz_audio/<video>.wav
        zzz_subtitle/<video>.srt
        zzz_translation/<video>.json       # Store's per-video file
        zzz_official_translation/<video>.srt
        zzz_material/<video>.pdf
        zzz_markdown/<video>.md
        zzz_zip/<video>.zip
        zzz_working/<video>.work
        metadata.json                      # Store's course-level file

Three-level key matching (O(1) per lookup)::

    1. by_id          — files whose stem ends with [VIDEOID] (yt-dlp style)
    2. by_stem        — exact stem after stripping any trailing [id]
    3. stat fallback  — on miss, try ``<key><default_suffix>`` on disk
                        (catches files created after the initial index)

API — always returns ``list[Path]``; single-key queries are a special case::

    ws.home.files()                        # all (natsort)
    ws.home.files("lec03")                 # [] or [Path]
    ws.home.files(["lec03", "lec07"])      # batch; missing="skip"|"raise"|"none"
    ws.home.path_for("lec03", suffix=".mp4")  # synthesize write path

    ws.videos(include=None, exclude=None)  # list of video keys (home stems)
    ws.routes(video)                       # dict[subdir_call, Path|None]
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Mapping,
    Sequence,
)

try:
    from natsort import natsorted  # type: ignore
except ImportError:  # pragma: no cover — natsort is a hard dependency

    def natsorted(seq, key=None):  # type: ignore
        return sorted(seq, key=key)


# ---------------------------------------------------------------------------
# Key helpers — yt-dlp [id] parsing
# ---------------------------------------------------------------------------

# Matches a trailing "[VIDEOID]" (letters/digits/-/_, at least 6 chars).
_ID_RE = re.compile(r"\s*\[([A-Za-z0-9_\-]{6,})\]\s*$")

# Matches a trailing language tag like ".en", ".en-US", ".zh-CN".
_LANG_TAIL_RE = re.compile(r"\.([a-z]{2}(?:-[A-Za-z0-9]+)?)$")


def extract_id(stem: str) -> str | None:
    """Return the trailing ``[VIDEOID]`` of *stem*, if present.

    >>> extract_id("Intro [LUU0EuDKgKo]")
    'LUU0EuDKgKo'
    >>> extract_id("Intro") is None
    True
    """
    m = _ID_RE.search(stem)
    return m.group(1) if m else None


def strip_id(stem: str) -> str:
    """Remove a trailing ``[VIDEOID]`` (and surrounding whitespace) from *stem*.

    >>> strip_id("Intro [LUU0EuDKgKo]")
    'Intro'
    >>> strip_id("Intro")
    'Intro'
    """
    return _ID_RE.sub("", stem).rstrip()


def strip_lang_tail(stem: str) -> str:
    """Strip ``.en``/``.en-US`` style language tag from *stem*.

    Preserves any trailing ``[id]``.
    """
    m = _ID_RE.search(stem)
    if m:
        body = stem[: m.start()]
        tail = m.group(0)
        body = _LANG_TAIL_RE.sub("", body.rstrip())
        return f"{body}{tail}" if body else tail.strip()
    return _LANG_TAIL_RE.sub("", stem)


def canonical_key(key: str | Path) -> str:
    """Return a matching-friendly stem for *key*.

    Accepts a plain stem, a file name, or a Path. Strips the extension.
    Does *not* alter whitespace or punctuation — we match exact stems.
    """
    if isinstance(key, Path):
        return key.stem
    s = str(key)
    if s.endswith(tuple(f"{sep}" for sep in (os.sep, "/"))):  # pragma: no cover
        raise ValueError(f"key must not be a directory: {s!r}")
    # Accept both "name" and "name.ext"; store stem.
    p = Path(s)
    return p.stem if p.suffix else s


# ---------------------------------------------------------------------------
# SubDir spec + registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubDirSpec:
    """Static declaration of one subdirectory in the course layout.

    Attributes:
        call: Attribute name on ``Workspace`` (e.g. ``"translation"``).
        name: Relative directory under ``<root>/<course>/`` (e.g.
            ``"zzz_translation"``). Use ``""`` for the course root
            itself (the "home" subdir).
        types: Allowed file extensions (e.g. ``(".mp4", ".mkv")``).
            Empty tuple = any extension.
        default_suffix: Extension used by ``path_for`` when the caller
            does not pass ``suffix=``. Also used by stat fallback.
        truncate_stem: If set, limit stems to this many characters on write.
            Used by Bilibili subdirs (limit 80) to satisfy filesystem caps
            while preserving any trailing ``[id]``.
        strip_id_on_write: If True, strip the trailing ``[id]`` from the
            stem before writing. Lookup still matches by id (the id was
            indexed from the original download filename elsewhere).
    """

    call: str
    name: str
    types: tuple[str, ...] = ()
    default_suffix: str | None = None
    truncate_stem: int | None = None
    strip_id_on_write: bool = False


# Ordered registry (insertion order). Global, process-wide.
_REGISTRY: list[SubDirSpec] = []
_REGISTRY_LOCK = threading.Lock()


def register_subdir(
    *,
    call: str,
    name: str,
    types: Sequence[str] = (),
    default_suffix: str | None = None,
    truncate_stem: int | None = None,
    strip_id_on_write: bool = False,
) -> SubDirSpec:
    """Register a subdirectory spec. Idempotent on ``call``.

    Adding a new directory type is a one-liner::

        register_subdir(call="audio", name="zzz_audio",
                        types=(".wav", ".mp3"), default_suffix=".wav")

    Returns the stored ``SubDirSpec`` (useful for tests).
    """
    spec = SubDirSpec(
        call=call,
        name=name,
        types=tuple(types),
        default_suffix=default_suffix,
        truncate_stem=truncate_stem,
        strip_id_on_write=strip_id_on_write,
    )
    with _REGISTRY_LOCK:
        for i, existing in enumerate(_REGISTRY):
            if existing.call == call:
                _REGISTRY[i] = spec
                return spec
        _REGISTRY.append(spec)
    return spec


def registered_specs() -> tuple[SubDirSpec, ...]:
    """Return a snapshot of the currently registered specs (for tests/debug)."""
    with _REGISTRY_LOCK:
        return tuple(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in subdirs
# ---------------------------------------------------------------------------
# Registered at import time. Adding a directory type for a new feature is
# one ``register_subdir`` call — no Workspace change required.

# Home: videos sit directly at <root>/<course>/ (no zzz_ prefix).
register_subdir(
    call="home",
    name="",
    types=(".mp4", ".mkv", ".mov", ".ts", ".flv", ".wmv", ".webm", ".m4a", ".wav"),
    default_suffix=".mp4",
)
register_subdir(
    call="audio",
    name="zzz_audio",
    types=(".wav", ".mp3", ".m4a", ".flac"),
    default_suffix=".wav",
)
register_subdir(
    call="subtitle",
    name="zzz_subtitle",
    types=(".srt", ".vtt", ".json"),
    default_suffix=".srt",
)
register_subdir(
    call="translation",
    name="zzz_translation",
    types=(".json",),
    default_suffix=".json",
)
# Sidecar raw_segment jsonl files (D-069). File stems look like
# ``<video>.words.jsonl`` or ``<video>.segments.jsonl`` — two-part suffix;
# we don't default-index/stat-lookup these, callers pass the full suffix.
register_subdir(
    call="subtitle_jsonl",
    name="zzz_subtitle_jsonl",
    types=(".jsonl",),
)
register_subdir(
    call="official_translation",
    name="zzz_official_translation",
    types=(".srt", ".vtt"),
    default_suffix=".srt",
)
register_subdir(
    call="material",
    name="zzz_material",
    types=(".pdf", ".pptx", ".md", ".txt"),
)
register_subdir(
    call="markdown",
    name="zzz_markdown",
    types=(".md",),
    default_suffix=".md",
)
register_subdir(
    call="zip",
    name="zzz_zip",
    types=(".zip",),
    default_suffix=".zip",
)
register_subdir(
    call="working",
    name="zzz_working",
)


# ---------------------------------------------------------------------------
# SubDir
# ---------------------------------------------------------------------------


class SubDir:
    """One subdirectory of a course, with O(1) key routing.

    Always returns ``list[Path]`` from ``files()`` — single-key lookup is
    just a list of length 0 or 1. This makes batch and single code paths
    identical.
    """

    __slots__ = (
        "spec",
        "workspace",
        "path",
        "_by_id",
        "_by_stem",
        "_all",
    )

    def __init__(self, workspace: "Workspace", spec: SubDirSpec) -> None:
        self.spec = spec
        self.workspace = workspace
        self.path = workspace.root / workspace.course if spec.name == "" else workspace.root / workspace.course / spec.name
        self._by_id: dict[str, Path] = {}
        self._by_stem: dict[str, Path] = {}
        self._all: list[Path] = []

    # -- read ------------------------------------------------------------

    def files(
        self,
        keys: str | Path | Iterable[str | Path] | None = None,
        *,
        missing: str = "skip",
    ) -> list[Path | None]:
        """Return files matching *keys*. Always a list.

        - ``keys=None`` — every indexed file (natsort by stem).
        - ``keys="x"`` or ``keys=Path("x.mp4")`` — single-key query,
          returns ``[]`` (missing="skip"), ``[None]`` (missing="none"),
          or raises ``FileNotFoundError`` (missing="raise").
        - ``keys=iterable`` — batch; each key resolved independently.
          ``missing="none"`` preserves positional alignment with *keys*
          (useful for cross-subdir joins).
        """
        if missing not in ("skip", "raise", "none"):
            raise ValueError(f"missing must be 'skip'|'raise'|'none', got {missing!r}")
        if keys is None:
            return natsorted(list(self._all), key=lambda p: p.stem)
        if isinstance(keys, (str, Path)):
            keys_list: list[str | Path] = [keys]
        else:
            keys_list = list(keys)

        result: list[Path | None] = []
        for k in keys_list:
            p = self._resolve_one(k)
            if p is None:
                if missing == "raise":
                    raise FileNotFoundError(f"{self.spec.call}: no file for key {k!r} under {self.path}")
                if missing == "none":
                    result.append(None)
                # "skip" → drop; validation already done above
            else:
                result.append(p)
        return result

    # -- write -----------------------------------------------------------

    def path_for(
        self,
        key: str | Path,
        *,
        suffix: str | None = None,
    ) -> Path:
        """Synthesize the write destination for *key*.

        Honors spec flags:

        * ``strip_id_on_write`` — drop the trailing ``[id]`` from the stem
          (still matchable by id on future reads).
        * ``truncate_stem`` — cap stem length (Bilibili 80-char limit).
          If an ``[id]`` is present and we are *not* stripping, the id is
          preserved and the body is truncated to fit.

        ``suffix`` overrides ``spec.default_suffix``.
        """
        stem = canonical_key(key)
        raw_id = extract_id(stem)

        if self.spec.strip_id_on_write:
            stem = strip_id(stem)
            if self.spec.truncate_stem is not None:
                stem = stem[: self.spec.truncate_stem]
        elif self.spec.truncate_stem is not None:
            limit = self.spec.truncate_stem
            if raw_id:
                id_part = f" [{raw_id}]"
                body = strip_id(stem)
                body_max = max(0, limit - len(id_part))
                stem = f"{body[:body_max].rstrip()}{id_part}"
            else:
                stem = stem[:limit]

        ext = suffix if suffix is not None else (self.spec.default_suffix or "")
        return self.path / f"{stem}{ext}"

    # -- index management ------------------------------------------------

    def _rebuild_index(self) -> None:
        self._by_id.clear()
        self._by_stem.clear()
        self._all.clear()
        if not self.path.exists():
            return
        # Use iterdir (non-recursive): home dir must ignore zzz_* children,
        # and we don't want nested trash interfering.
        try:
            entries = list(self.path.iterdir())
        except OSError:
            return
        for p in entries:
            if not p.is_file():
                continue
            if self.spec.name == "" and p.name == "metadata.json":
                # Reserved for Store's course-level file.
                continue
            if self.spec.types and p.suffix not in self.spec.types:
                continue
            self._add_to_index(p)

    def _add_to_index(self, p: Path) -> None:
        stem = p.stem
        self._all.append(p)
        id_ = extract_id(stem)
        if id_:
            self._by_id[id_] = p
        self._by_stem[strip_id(stem)] = p

    def _resolve_one(self, key: str | Path) -> Path | None:
        stem = canonical_key(key)
        # 1) exact-id match
        id_ = extract_id(stem)
        if id_:
            hit = self._by_id.get(id_)
            if hit is not None:
                return hit
        # 2) stem match (after id-stripping both sides)
        stripped = strip_id(stem)
        hit = self._by_stem.get(stripped)
        if hit is not None:
            return hit
        # 3) stat fallback — catch files created after the initial walk
        if self.spec.default_suffix is not None:
            candidate = self.path / f"{stripped}{self.spec.default_suffix}"
            if candidate.is_file():
                self._add_to_index(candidate)
                return candidate
            if stripped != stem:
                candidate = self.path / f"{stem}{self.spec.default_suffix}"
                if candidate.is_file():
                    self._add_to_index(candidate)
                    return candidate
        return None

    # -- misc ------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SubDir(call={self.spec.call!r}, path={str(self.path)!r}, n={len(self._all)})"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """A course-rooted file layout.

    ``Workspace(root, course)`` walks the course tree once at construction
    and builds O(1) indexes for every registered subdirectory. Subsequent
    ``files()`` / ``path_for()`` calls are constant-time.

    New files created during a run are picked up by the stat fallback on
    lookup; ``reindex()`` re-walks everything if you need a hard reset.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        course: str,
        *,
        specs: Sequence[SubDirSpec] | None = None,
    ) -> None:
        if not course or "/" in course and ".." in Path(course).parts:
            raise ValueError(f"invalid course name: {course!r}")
        self.root = Path(root)
        self.course = course
        self._subdirs: dict[str, SubDir] = {}
        # Snapshot the registry so later registrations don't leak into an
        # already-constructed Workspace.
        active_specs = tuple(specs) if specs is not None else registered_specs()
        for spec in active_specs:
            sd = SubDir(self, spec)
            self._subdirs[spec.call] = sd
            sd._rebuild_index()
        # Expose each SubDir as a plain attribute for discoverability.
        for call, sd in self._subdirs.items():
            if hasattr(self, call):
                raise ValueError(f"subdir call name {call!r} collides with Workspace attribute")
            object.__setattr__(self, call, sd)

    # -- lookup ----------------------------------------------------------

    @property
    def course_path(self) -> Path:
        return self.root / self.course

    @property
    def metadata_path(self) -> Path:
        """Path to the course-level metadata.json (Store's domain)."""
        return self.course_path / "metadata.json"

    def subdirs(self) -> Mapping[str, SubDir]:
        return dict(self._subdirs)

    def get_subdir(self, call: str) -> SubDir:
        try:
            return self._subdirs[call]
        except KeyError as e:
            raise AttributeError(f"no subdir registered for call={call!r}") from e

    # -- cross-subdir queries -------------------------------------------

    def videos(
        self,
        *,
        include: Iterable[str] | Callable[[str], bool] | None = None,
        exclude: Iterable[str] | Callable[[str], bool] | None = None,
    ) -> list[str]:
        """List video keys under this course (home subdir stems, natsort).

        ``include`` and ``exclude`` accept either an iterable of exact keys
        or a callable predicate. Predicates receive the stem; iterables
        match equality (after stripping trailing [id] on the caller side
        if they want — we don't guess).

        Typical failure-rerun pattern::

            pending = ws.videos(exclude=lambda v: ws.translation.files(v))
            for v in pending:
                process(v)
        """
        home = self.get_subdir("home")
        stems = [p.stem for p in home.files()]

        def _match(pred, stem: str) -> bool:
            if callable(pred):
                return bool(pred(stem))
            try:
                return stem in pred  # set/list
            except TypeError:  # pragma: no cover
                return False

        if include is not None:
            stems = [s for s in stems if _match(include, s)]
        if exclude is not None:
            stems = [s for s in stems if not _match(exclude, s)]
        return stems

    def routes(self, video: str | Path) -> dict[str, Path | None]:
        """Return a map ``{subdir_call: resolved_path_or_None}`` for *video*.

        One line to see every artifact produced for a video across the pipeline.
        """
        out: dict[str, Path | None] = {}
        for call, sd in self._subdirs.items():
            hits = sd.files(video, missing="none")
            out[call] = hits[0] if hits else None
        return out

    # -- course-level ----------------------------------------------------

    @classmethod
    def courses(cls, root: str | os.PathLike[str]) -> list[str]:
        """List course directory names under *root* (natsort)."""
        root_path = Path(root)
        if not root_path.is_dir():
            return []
        try:
            entries = [p.name for p in root_path.iterdir() if p.is_dir()]
        except OSError:  # pragma: no cover
            return []
        return natsorted(entries)

    def reindex(self) -> None:
        """Re-walk every subdir from scratch. Cheap; called rarely."""
        for sd in self._subdirs.values():
            sd._rebuild_index()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Workspace(root={str(self.root)!r}, course={self.course!r})"
