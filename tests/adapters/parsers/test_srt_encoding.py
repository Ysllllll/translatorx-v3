"""C10 — SrtSource read_srt encoding sniffing.

Real-world SRTs ship with mixed encodings: BOM-marked UTF-8/16, GBK
(``cp936``), Shift-JIS, Windows-1252. The reader must transparently
decode all of them without raising ``UnicodeDecodeError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.parsers import read_srt


SAMPLE_BY_ENC = {"utf-8": "1\n00:00:00,000 --> 00:00:01,000\n你好世界 こんにちは\n", "utf-8-sig": "1\n00:00:00,000 --> 00:00:01,000\n你好世界 こんにちは\n", "gb18030": "1\n00:00:00,000 --> 00:00:01,000\n你好世界\n", "shift_jis": "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n"}


@pytest.mark.parametrize("encoding", list(SAMPLE_BY_ENC.keys()))
def test_read_srt_handles_common_encodings(tmp_path: Path, encoding: str) -> None:
    p = tmp_path / f"sample-{encoding}.srt"
    p.write_bytes(SAMPLE_BY_ENC[encoding].encode(encoding))
    segs = read_srt(p)
    assert len(segs) == 1
    assert segs[0].start == 0.0


def test_read_srt_utf16_with_bom(tmp_path: Path) -> None:
    p = tmp_path / "u16.srt"
    p.write_bytes(SAMPLE_BY_ENC["utf-8"].encode("utf-16"))
    segs = read_srt(p)
    assert len(segs) == 1
    assert "你好" in segs[0].text


def test_read_srt_falls_back_on_garbage(tmp_path: Path) -> None:
    """Even unparseable bytes shouldn't raise — replacement decoding wins."""
    p = tmp_path / "garbage.srt"
    p.write_bytes(b"1\n00:00:00,000 --> 00:00:01,000\nhello \xff\xfe\xab\xcd\n")
    segs = read_srt(p)
    assert len(segs) == 1
