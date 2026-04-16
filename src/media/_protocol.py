"""Media source protocol and data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MediaInfo:
    """Metadata for a single media item (video/audio)."""

    id: str
    title: str
    url: str
    platform: str = ""
    duration: float = 0.0
    subtitle_languages: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.title} [{self.id}]"


@dataclass(frozen=True)
class PlaylistInfo:
    """Metadata for a playlist / course / multi-part video."""

    id: str
    title: str
    url: str
    platform: str = ""
    entries: tuple[MediaInfo, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


@dataclass(frozen=True)
class DownloadResult:
    """Result of a download operation."""

    path: Path
    media_info: MediaInfo


@runtime_checkable
class MediaSource(Protocol):
    """Protocol for media acquisition from online platforms.

    Implementations fetch metadata and download audio/video/subtitles
    from a specific platform or via a universal backend (e.g. yt-dlp).
    """

    def get_info(self, url: str) -> MediaInfo | PlaylistInfo:
        """Fetch metadata for a URL (single video or playlist).

        Returns ``MediaInfo`` for a single video, ``PlaylistInfo`` for
        a playlist or multi-part video.
        """
        ...

    def download_audio(
        self,
        url: str,
        output_dir: Path,
        *,
        cookies: str | Path | None = None,
    ) -> DownloadResult:
        """Download audio track to *output_dir*.

        Returns the path to the downloaded audio file (m4a/mp3/wav).
        """
        ...

    def download_subtitle(
        self,
        url: str,
        output_dir: Path,
        *,
        language: str = "en",
        cookies: str | Path | None = None,
    ) -> DownloadResult | None:
        """Download platform subtitles if available.

        Returns ``None`` when no subtitle in the requested language exists.
        """
        ...

    def download_video(
        self,
        url: str,
        output_dir: Path,
        *,
        cookies: str | Path | None = None,
    ) -> DownloadResult:
        """Download video file to *output_dir*."""
        ...


@runtime_checkable
class MediaProbe(Protocol):
    """Protocol for probing local media files."""

    def probe(self, path: Path) -> MediaFileInfo:
        """Return technical metadata for a local media file."""
        ...


@dataclass(frozen=True)
class MediaFileInfo:
    """Technical metadata for a local audio/video file."""

    duration: float = 0.0
    sample_rate: int = 0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    has_video: bool = False

    @property
    def is_audio_only(self) -> bool:
        return self.has_audio and not self.has_video
