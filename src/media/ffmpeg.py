"""FFmpeg-based media operations: probing and audio extraction."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .protocol import MediaFileInfo


def _run_ffprobe(path: Path) -> dict:
    """Run ffprobe and return the parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json

    return json.loads(result.stdout)


def probe(path: Path) -> MediaFileInfo:
    """Probe a local media file and return its technical metadata.

    Raises ``FileNotFoundError`` if *path* does not exist.
    Raises ``subprocess.CalledProcessError`` if ffprobe fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = _run_ffprobe(path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    return MediaFileInfo(
        duration=float(fmt.get("duration", 0.0)),
        sample_rate=int(audio_stream["sample_rate"]) if audio_stream and "sample_rate" in audio_stream else 0,
        width=int(video_stream["width"]) if video_stream and "width" in video_stream else 0,
        height=int(video_stream["height"]) if video_stream and "height" in video_stream else 0,
        has_audio=audio_stream is not None,
        has_video=video_stream is not None,
    )


def extract_audio(
    video_path: Path,
    output_path: Path | None = None,
    *,
    format: str = "m4a",
) -> Path:
    """Extract audio track from a video file.

    If *output_path* is ``None``, writes to ``<video_stem>.<format>`` in the
    same directory as the video.

    Returns the path to the extracted audio file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")
    output_path = Path(output_path)

    codec_map = {
        "m4a": "copy",
        "mp3": "libmp3lame",
        "wav": "pcm_s16le",
    }
    codec = codec_map.get(format, "copy")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-acodec",
        codec,
        "-y",  # overwrite
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
