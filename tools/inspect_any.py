"""Auto-route inspect tool: dispatches to inspect_srt / inspect_whisperx by extension.

Usage::

    python tools/inspect.py path/to/file.srt [args...]   # → inspect_srt
    python tools/inspect.py path/to/file.json [args...]  # → inspect_whisperx

All extra args are passed through to the routed tool.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0

    target = Path(sys.argv[1])
    ext = target.suffix.lower()

    if ext == ".srt":
        sys.argv = ["inspect_srt.py"] + sys.argv[1:]
        from inspect_srt import main as _main  # noqa: WPS433
    elif ext in (".json", ".jsonl"):
        sys.argv = ["inspect_whisperx.py"] + sys.argv[1:]
        from inspect_whisperx import main as _main  # noqa: WPS433
    else:
        print(f"[ERR] unrecognized extension: {ext!r}. Expected .srt or .json", file=sys.stderr)
        print("Use inspect_srt.py / inspect_whisperx.py directly to override.", file=sys.stderr)
        return 2

    return _main()


if __name__ == "__main__":
    # Make sibling modules importable
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
