"""Tests for media.ytdlp (YtdlpSource).

Most tests require network access and are skipped in CI.
Unit tests for parsing helpers run without network.
"""

import pytest

from adapters.media import MediaInfo, PlaylistInfo
from adapters.media.ytdlp import (
    _detect_platform,
    _extract_subtitle_languages,
    _parse_info,
    _parse_single,
)


class TestDetectPlatform:
    def test_youtube(self):
        assert _detect_platform({"extractor": "youtube", "webpage_url": ""}) == "youtube"
        assert _detect_platform({"extractor": "", "webpage_url": "https://www.youtube.com/watch?v=abc"}) == "youtube"

    def test_bilibili(self):
        assert _detect_platform({"extractor": "BiliBili", "webpage_url": ""}) == "bilibili"
        assert _detect_platform({"extractor": "", "webpage_url": "https://www.bilibili.com/video/BV1xx"}) == "bilibili"

    def test_unknown(self):
        assert _detect_platform({"extractor": "", "webpage_url": ""}) == "unknown"

    def test_other_extractor(self):
        assert _detect_platform({"extractor": "vimeo", "webpage_url": ""}) == "vimeo"


class TestExtractSubtitleLanguages:
    def test_empty(self):
        assert _extract_subtitle_languages({}) == ()

    def test_manual_subs(self):
        data = {"subtitles": {"en": [], "zh": []}}
        langs = _extract_subtitle_languages(data)
        assert "en" in langs
        assert "zh" in langs

    def test_auto_captions(self):
        data = {"automatic_captions": {"en": [], "ja": []}}
        langs = _extract_subtitle_languages(data)
        assert "en" in langs
        assert "ja" in langs

    def test_merged(self):
        data = {
            "subtitles": {"en": []},
            "automatic_captions": {"en": [], "fr": []},
        }
        langs = _extract_subtitle_languages(data)
        assert set(langs) == {"en", "fr"}


class TestParseSingle:
    def test_basic(self):
        data = {
            "id": "abc123",
            "title": "Test Video",
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
            "extractor": "youtube",
            "duration": 120.5,
            "subtitles": {"en": []},
        }
        info = _parse_single(data)
        assert isinstance(info, MediaInfo)
        assert info.id == "abc123"
        assert info.title == "Test Video"
        assert info.platform == "youtube"
        assert info.duration == 120.5
        assert "en" in info.subtitle_languages

    def test_missing_fields(self):
        info = _parse_single({})
        assert info.id == ""
        assert info.title == ""
        assert info.duration == 0.0


class TestParseInfo:
    def test_single_video(self):
        data = {
            "id": "v1",
            "title": "Single",
            "webpage_url": "https://youtube.com/watch?v=v1",
            "extractor": "youtube",
            "duration": 60,
        }
        result = _parse_info(data)
        assert isinstance(result, MediaInfo)

    def test_playlist(self):
        data = {
            "id": "pl1",
            "title": "Playlist",
            "webpage_url": "https://youtube.com/playlist?list=pl1",
            "extractor": "youtube",
            "entries": [
                {
                    "id": "v1",
                    "title": "Video 1",
                    "webpage_url": "https://youtube.com/watch?v=v1",
                    "extractor": "youtube",
                    "duration": 30,
                },
                None,  # yt-dlp sometimes returns None for unavailable entries
                {
                    "id": "v2",
                    "title": "Video 2",
                    "webpage_url": "https://youtube.com/watch?v=v2",
                    "extractor": "youtube",
                    "duration": 45,
                },
            ],
        }
        result = _parse_info(data)
        assert isinstance(result, PlaylistInfo)
        assert len(result) == 2  # None entry is filtered out
        assert result.entries[0].id == "v1"
        assert result.entries[1].id == "v2"

    def test_empty_playlist(self):
        data = {
            "id": "pl",
            "title": "Empty",
            "webpage_url": "",
            "extractor": "",
            "entries": [],
        }
        result = _parse_info(data)
        assert isinstance(result, PlaylistInfo)
        assert len(result) == 0
