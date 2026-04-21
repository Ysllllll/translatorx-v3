import unittest
from pathlib import Path


def resolve_test_font_path() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise unittest.SkipTest("no usable font file for plength test")


def expected_pixel_length(text: str, font: str, font_size: int) -> int:
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise unittest.SkipTest("Pillow is required for plength tests") from exc

    left, _, right, _ = ImageFont.truetype(font, font_size).getbbox(text)
    return max(0, int(right - left))


TEST_FONT_PATH = resolve_test_font_path()
