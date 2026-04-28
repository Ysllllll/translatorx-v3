"""Tests for PixelWidthRule — optional Pillow-based hallucination check."""

import pytest

from application.checker import Checker, PixelWidthLimits, PixelWidthRule, Severity, default_checker


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _find_font() -> str | None:
    candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    from pathlib import Path

    for p in candidates:
        if Path(p).exists():
            return p
    return None


class TestPixelWidthRule:
    def test_no_font_no_op(self):
        r = PixelWidthRule(limits=PixelWidthLimits(font_path=""))
        assert r.check("hello", "x" * 200) == []

    def test_invalid_font_no_op(self):
        r = PixelWidthRule(limits=PixelWidthLimits(font_path="/nonexistent/font.ttf"))
        assert r.check("hello", "x" * 200) == []

    @pytest.mark.skipif(not _has_pillow(), reason="Pillow not installed")
    def test_with_real_font_detects_hallucination(self):
        font_path = _find_font()
        if font_path is None:
            pytest.skip("No system font available")
        r = PixelWidthRule(limits=PixelWidthLimits(font_path=font_path, font_size=16, max_ratio=2.0))
        issues = r.check("Hello", "你好" * 50)
        assert any(i.rule == "pixel_width" for i in issues)

    @pytest.mark.skipif(not _has_pillow(), reason="Pillow not installed")
    def test_with_real_font_normal_passes(self):
        font_path = _find_font()
        if font_path is None:
            pytest.skip("No system font available")
        r = PixelWidthRule(limits=PixelWidthLimits(font_path=font_path, max_ratio=4.0))
        assert r.check("Hello world", "你好世界") == []

    def test_default_checker_includes_pixel_width(self):
        c = default_checker("en", "zh")
        names = [r.name for r in c.rules]
        assert "pixel_width" in names
