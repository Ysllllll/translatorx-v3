"""Tests for media.protocol data types."""

from adapters.media import (
    DownloadResult,
    MediaFileInfo,
    MediaInfo,
    MediaSource,
    PlaylistInfo,
)
from pathlib import Path


class TestMediaInfo:
    def test_frozen(self):
        info = MediaInfo(id="abc", title="Test", url="https://example.com")
        try:
            info.id = "xyz"  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass

    def test_display_name(self):
        info = MediaInfo(id="abc123", title="My Video", url="")
        assert info.display_name == "My Video [abc123]"

    def test_defaults(self):
        info = MediaInfo(id="x", title="T", url="u")
        assert info.platform == ""
        assert info.duration == 0.0
        assert info.subtitle_languages == ()


class TestPlaylistInfo:
    def test_len_and_iter(self):
        entries = (
            MediaInfo(id="1", title="A", url=""),
            MediaInfo(id="2", title="B", url=""),
        )
        pl = PlaylistInfo(id="pl", title="Playlist", url="", entries=entries)
        assert len(pl) == 2
        assert list(pl) == list(entries)

    def test_empty(self):
        pl = PlaylistInfo(id="pl", title="Empty", url="")
        assert len(pl) == 0
        assert list(pl) == []


class TestDownloadResult:
    def test_frozen(self):
        info = MediaInfo(id="x", title="T", url="u")
        result = DownloadResult(path=Path("/tmp/test.m4a"), media_info=info)
        assert result.path == Path("/tmp/test.m4a")
        assert result.media_info.id == "x"


class TestMediaFileInfo:
    def test_is_audio_only(self):
        audio = MediaFileInfo(has_audio=True, has_video=False)
        assert audio.is_audio_only is True

        video = MediaFileInfo(has_audio=True, has_video=True)
        assert video.is_audio_only is False


class TestProtocolConformance:
    """Verify that YtdlpSource satisfies the MediaSource protocol."""

    def test_ytdlp_is_media_source(self):
        from adapters.media import YtdlpSource

        assert isinstance(YtdlpSource(), MediaSource)
