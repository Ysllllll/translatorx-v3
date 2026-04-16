"""Media acquisition: download audio/video/subtitles from online platforms.

Public API
----------
Data types:
    MediaInfo, PlaylistInfo, DownloadResult, MediaFileInfo

Protocols:
    MediaSource, MediaProbe

Implementations:
    YtdlpSource  — yt-dlp backend (YouTube, Bilibili, etc.)

FFmpeg utilities:
    probe, extract_audio
"""

from .protocol import (
    DownloadResult,
    MediaFileInfo,
    MediaInfo,
    MediaProbe,
    MediaSource,
    PlaylistInfo,
)
from .ffmpeg import extract_audio, probe
from .ytdlp import YtdlpSource

__all__ = [
    # Data types
    "MediaInfo",
    "PlaylistInfo",
    "DownloadResult",
    "MediaFileInfo",
    # Protocols
    "MediaSource",
    "MediaProbe",
    # Implementations
    "YtdlpSource",
    # FFmpeg utilities
    "probe",
    "extract_audio",
]
