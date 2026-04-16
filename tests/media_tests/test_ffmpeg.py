"""Tests for media._ffmpeg (probe and extract_audio).

These tests require ffmpeg to be installed. They are skipped otherwise.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from media import extract_audio, probe

ffmpeg_available = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available, reason="ffmpeg not installed"
)


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Generate a short silent audio file for testing."""
    path = tmp_path / "test.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "0.5",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Generate a short silent video with audio for testing."""
    path = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=0.5",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", "0.5",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


@requires_ffmpeg
class TestProbe:
    def test_probe_audio(self, sample_audio: Path):
        info = probe(sample_audio)
        assert info.has_audio is True
        assert info.has_video is False
        assert info.is_audio_only is True
        assert info.sample_rate == 44100
        assert info.duration > 0

    def test_probe_video(self, sample_video: Path):
        info = probe(sample_video)
        assert info.has_audio is True
        assert info.has_video is True
        assert info.is_audio_only is False
        assert info.width == 320
        assert info.height == 240

    def test_probe_not_found(self):
        with pytest.raises(FileNotFoundError):
            probe(Path("/nonexistent/file.mp4"))


@requires_ffmpeg
class TestExtractAudio:
    def test_extract_default(self, sample_video: Path):
        audio_path = extract_audio(sample_video)
        assert audio_path.exists()
        assert audio_path.suffix == ".m4a"
        info = probe(audio_path)
        assert info.has_audio is True

    def test_extract_custom_output(self, sample_video: Path, tmp_path: Path):
        out = tmp_path / "custom_audio.mp3"
        result = extract_audio(sample_video, out, format="mp3")
        assert result == out
        assert out.exists()

    def test_extract_wav(self, sample_video: Path, tmp_path: Path):
        out = tmp_path / "audio.wav"
        result = extract_audio(sample_video, out, format="wav")
        assert result.exists()
        info = probe(result)
        assert info.has_audio is True

    def test_extract_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_audio(Path("/nonexistent/video.mp4"))
