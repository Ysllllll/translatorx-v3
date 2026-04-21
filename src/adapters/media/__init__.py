"""Media acquisition adapters — yt-dlp + ffmpeg."""

from ports.media import (
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
    "DownloadResult",
    "MediaFileInfo",
    "MediaInfo",
    "MediaProbe",
    "MediaSource",
    "PlaylistInfo",
    "YtdlpSource",
    "extract_audio",
    "probe",
]
