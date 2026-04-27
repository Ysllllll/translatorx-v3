"""yt-dlp based MediaSource implementation.

Handles any platform supported by yt-dlp (YouTube, Bilibili, etc.).
Platform-specific behaviour (cookies, format preferences) is driven
by configuration, not by separate classes.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yt_dlp

from ports.media import DownloadResult, MediaInfo, PlaylistInfo


def _find_downloaded_file(directory: Path, video_id: str) -> Path | None:
    """Find a file in *directory* whose name contains ``[video_id]``."""
    for p in directory.iterdir():
        if f"[{video_id}]" in p.name and not p.name.endswith(".part"):
            return p
    return None


def _make_outtmpl(playlist_index: int | None = None) -> str:
    if playlist_index is not None and playlist_index > 0:
        return rf"P{playlist_index} %(title)s [%(id)s].%(ext)s"
    return r"%(title)s [%(id)s].%(ext)s"


def _base_opts(cookies: str | Path | None = None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "ignoreerrors": True,
        "fragment_retries": 10,
        "retries": 10,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies is not None:
        opts["cookiefile"] = str(cookies)
    return opts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class YtdlpSource:
    """MediaSource backed by yt-dlp.

    Works with any yt-dlp-supported platform.  Pass *cookies* for sites
    that require authentication (e.g. Bilibili).
    """

    cookies: str | Path | None = None

    # -- info ---------------------------------------------------------------

    def get_info(self, url: str) -> MediaInfo | PlaylistInfo:
        """Fetch metadata without downloading."""
        opts = {**_base_opts(self.cookies), "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)

        if data is None:
            raise RuntimeError(f"yt-dlp returned no info for {url}")

        return _parse_info(data)

    # -- download audio -----------------------------------------------------

    def download_audio(
        self,
        url: str,
        output_dir: Path,
        *,
        cookies: str | Path | None = None,
    ) -> DownloadResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        info = self._extract_single(url)
        video_id = info.id

        existing = _find_downloaded_file(output_dir, video_id)
        if existing is not None:
            return DownloadResult(path=existing, media_info=info)

        ck = cookies or self.cookies
        opts = {
            **_base_opts(ck),
            "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
            "outtmpl": _make_outtmpl(),
            "paths": {"home": str(output_dir)},
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        downloaded = _find_downloaded_file(output_dir, video_id)
        if downloaded is None:
            raise RuntimeError(f"Audio download failed for {url}")
        return DownloadResult(path=downloaded, media_info=info)

    # -- download subtitle --------------------------------------------------

    def download_subtitle(
        self,
        url: str,
        output_dir: Path,
        *,
        language: str = "en",
        cookies: str | Path | None = None,
    ) -> DownloadResult | None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        info = self._extract_single(url)

        # Check if requested language is available
        if language not in info.subtitle_languages:
            return None

        ck = cookies or self.cookies
        opts = {
            **_base_opts(ck),
            "writesubtitles": True,
            "writeautomaticsub": False,
            "skip_download": True,
            "subtitlesformat": "srt/best",
            "subtitleslangs": [language],
            "outtmpl": _make_outtmpl(),
            "paths": {"home": str(output_dir)},
            "postprocessors": [
                {
                    "key": "FFmpegSubtitlesConvertor",
                    "format": "srt",
                    "when": "after_dl",
                }
            ],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        # yt-dlp names subtitle files with language suffix
        for p in output_dir.iterdir():
            if f"[{info.id}]" in p.name and p.suffix == ".srt":
                return DownloadResult(path=p, media_info=info)

        return None

    # -- download video -----------------------------------------------------

    def download_video(
        self,
        url: str,
        output_dir: Path,
        *,
        cookies: str | Path | None = None,
        max_height: int = 1080,
    ) -> DownloadResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        info = self._extract_single(url)
        video_id = info.id

        existing = _find_downloaded_file(output_dir, video_id)
        if existing is not None:
            return DownloadResult(path=existing, media_info=info)

        ck = cookies or self.cookies
        opts = {
            **_base_opts(ck),
            "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]",
            "merge_output_format": "mkv",
            "outtmpl": _make_outtmpl(),
            "paths": {"home": str(output_dir)},
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        downloaded = _find_downloaded_file(output_dir, video_id)
        if downloaded is None:
            raise RuntimeError(f"Video download failed for {url}")
        return DownloadResult(path=downloaded, media_info=info)

    # -- internals ----------------------------------------------------------

    def _extract_single(self, url: str) -> MediaInfo:
        """Extract info for a single video (not playlist)."""
        opts = {**_base_opts(self.cookies), "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)
        if data is None:
            raise RuntimeError(f"yt-dlp returned no info for {url}")

        result = _parse_info(data)
        if isinstance(result, PlaylistInfo) and result.entries:
            return result.entries[0]
        if isinstance(result, MediaInfo):
            return result
        raise RuntimeError(f"Cannot resolve single media from {url}")

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _detect_platform(data: dict) -> str:
    extractor = data.get("extractor", "").lower()
    url = data.get("webpage_url", "")
    if "youtube" in extractor or "youtube.com" in url:
        return "youtube"
    if "bilibili" in extractor or "bilibili.com" in url:
        return "bilibili"
    return extractor or "unknown"


def _extract_subtitle_languages(data: dict) -> tuple[str, ...]:
    subs = data.get("subtitles", {})
    auto_subs = data.get("automatic_captions", {})
    all_langs = set(subs.keys()) | set(auto_subs.keys())
    return tuple(sorted(all_langs))


def _parse_single(data: dict) -> MediaInfo:
    return MediaInfo(
        id=data.get("id", ""),
        title=data.get("title", ""),
        url=data.get("webpage_url", data.get("url", "")),
        platform=_detect_platform(data),
        duration=float(data.get("duration", 0) or 0),
        subtitle_languages=_extract_subtitle_languages(data),
    )


def _parse_info(data: dict) -> MediaInfo | PlaylistInfo:
    entries = data.get("entries")
    if entries is None:
        return _parse_single(data)

    items: list[MediaInfo] = []
    for entry in entries:
        if entry is not None:
            items.append(_parse_single(entry))

    return PlaylistInfo(
        id=data.get("id", ""),
        title=data.get("title", ""),
        url=data.get("webpage_url", data.get("url", "")),
        platform=_detect_platform(data),
        entries=tuple(items),
    )
