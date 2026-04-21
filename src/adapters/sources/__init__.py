"""Source implementations — produce :class:`SentenceRecord` streams.

A :class:`~runtime.protocol.Source` is the front of a processor chain
(D-067). Sources read raw inputs (SRT files, WhisperX JSON, live
segment feeds) and yield enriched :class:`SentenceRecord` items with
auto-allocated ``extra["id"]`` so downstream processors can persist
via ``Store.patch_video`` using dotted keys.

Built-in sources:

* :class:`SrtSource` — parse an SRT file and stream sentence records.
* :class:`WhisperXSource` — parse a WhisperX JSON and stream sentence
  records.
* :class:`PushQueueSource` — receive segments via ``feed()`` / ``close()``
  and stream completed sentence records (for the browser-plugin / live
  scenario).
"""

from __future__ import annotations

from adapters.sources.push import PushQueueSource
from adapters.sources.srt import SrtSource
from adapters.sources.whisperx import WhisperXSource

__all__ = [
    "PushQueueSource",
    "SrtSource",
    "WhisperXSource",
]
