"""SRT serialization + content-invariant helpers."""

from __future__ import annotations

import re
import unicodedata

from .parse import _ms_to_ts
from .patterns import _HTML_ENTITY_RE, _HTML_TAG_RE, _INVISIBLE_RE, _entity_sub
from .types import Cue


def dump(cues: list[Cue]) -> str:
    """Serialize cues to a standard-shaped SRT string (F1-F5, E1-E5, N1)."""
    parts: list[str] = []
    for idx, c in enumerate(cues, start=1):
        parts.append(str(idx))
        parts.append(f"{_ms_to_ts(c.start_ms)} --> {_ms_to_ts(c.end_ms)}")
        parts.append(c.text)
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


_STRIP_PUNCT_RE = re.compile(r"[\s\u2000-\u206f\u3000-\u303f\uff00-\uffef" + re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""") + r"]+")


def text_content(cues_or_text: list[Cue] | str) -> str:
    """Extract content invariant: all text joined, spaces + punctuation removed."""
    if isinstance(cues_or_text, str):
        joined = cues_or_text
    else:
        joined = "".join(c.text for c in cues_or_text)
    joined = _HTML_ENTITY_RE.sub(_entity_sub, joined)
    joined = _HTML_TAG_RE.sub("", joined)
    joined = _INVISIBLE_RE.sub("", joined)
    joined = unicodedata.normalize("NFKC", joined)
    joined = _STRIP_PUNCT_RE.sub("", joined)
    joined = "".join(ch for ch in joined if unicodedata.category(ch)[0] != "P")
    return joined


__all__ = ["dump", "text_content", "_STRIP_PUNCT_RE"]
