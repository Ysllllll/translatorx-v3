"""Run all multilingual demos in sequence."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from demo_processing import main as processing_main  # noqa: E402
from demo_translate import main as translate_main  # noqa: E402
from demo_course import main as course_main  # noqa: E402


def main() -> None:
    processing_main()
    print()
    asyncio.run(translate_main())
    print()
    asyncio.run(course_main())


if __name__ == "__main__":
    main()
