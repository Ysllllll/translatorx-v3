"""TTSProcessor — synthesize target-language audio per record.

For each :class:`SentenceRecord`:

1. Pick the alignment pieces (``rec.alignment[target]``) — one piece per
   segment. When alignment is missing, fall back to the full
   ``translations[target]`` rendered against ``(rec.start, rec.end)``.
2. Resolve the voice via :class:`VoicePicker` (keyed by
   ``rec.segments[i].speaker``).
3. Call the :class:`TTS` backend to synthesize audio bytes.
4. Write ``<workspace>/zzz_tts/<video>.<rec_id>.<seg_idx>.<ext>`` to
   disk.
5. Record the relative audio paths under
   ``rec.extra["tts"][target]``.

The processor never modifies :class:`TranslationContext` or
``translations``; it only emits audio artifacts + bookkeeping into the
record's ``extra``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, AsyncIterator

from application.translate import TranslationContext
from domain.model import SentenceRecord

from ports.processor import ProcessorBase
from ports.tts import TTS, SynthesizeOptions, Voice, VoicePicker

if TYPE_CHECKING:
    from ports.source import VideoKey
    from adapters.storage.store import Store
    from application.orchestrator.session import VideoSession


logger = logging.getLogger(__name__)


class TTSProcessor(ProcessorBase[SentenceRecord, SentenceRecord]):
    """Render target-language audio for each record.

    Args:
        tts: :class:`TTS` backend instance.
        voice_picker: :class:`VoicePicker` that resolves
            ``rec.segments[i].speaker`` to a :class:`Voice`.
        default_voice: Voice used when the picker has no mapping and the
            backend returns no voices.
        format: Audio container (``"mp3"`` / ``"wav"`` / ...).
        rate: Global speech-rate multiplier (e.g. ``1.05`` when target
            language is verbose and needs to fit source timing).
        skip_if_exists: When ``True`` and the target audio file already
            exists on disk, reuse it rather than re-synthesizing.
    """

    name = "tts"

    def __init__(
        self,
        tts: TTS,
        *,
        voice_picker: VoicePicker | None = None,
        default_voice: Voice | str | None = None,
        format: str = "mp3",
        rate: float = 1.0,
        skip_if_exists: bool = True,
    ) -> None:
        self._tts = tts
        self._voice_picker = voice_picker or VoicePicker(default_voice=default_voice)
        self._format = format
        self._rate = rate
        self._skip_if_exists = skip_if_exists
        self._fp_cache: str | None = None

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        if self._fp_cache is not None:
            return self._fp_cache
        raw = f"backend={type(self._tts).__name__}|format={self._format}|rate={self._rate}"
        self._fp_cache = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._fp_cache

    # ------------------------------------------------------------------
    # process
    # ------------------------------------------------------------------

    async def process(
        self,
        upstream: AsyncIterator[SentenceRecord],
        *,
        ctx: TranslationContext,
        store: "Store",
        video_key: "VideoKey",
        session: "VideoSession | None" = None,
    ) -> AsyncIterator[SentenceRecord]:
        target = ctx.target_lang
        fp = self.fingerprint()

        workspace = getattr(store, "workspace", None)
        if workspace is None:
            raise RuntimeError(
                "TTSProcessor requires a Store with a `.workspace` attribute (Workspace layout needed to resolve audio paths)."
            )
        tts_subdir = workspace.get_subdir("tts")

        if session is None:
            from application.orchestrator.session import VideoSession  # noqa: PLC0415

            session = await VideoSession.load(store, video_key)
            owned_session = True
        else:
            owned_session = False

        try:
            async for rec in upstream:
                rec_id = rec.extra.get("id")

                translation = rec.get_translation(target, default_variant_key=ctx.variant.key)
                if not translation or not translation.strip():
                    yield rec
                    continue

                pieces, speakers = self._pieces_and_speakers(rec, target, translation)
                paths: list[str] = []

                for idx, (piece, speaker) in enumerate(zip(pieces, speakers)):
                    if not piece or not piece.strip():
                        paths.append("")
                        continue

                    stem = self._stem_for(video_key.video, rec_id, idx)
                    suffix = f".{self._format.lstrip('.')}"
                    out_path = tts_subdir.path_for(stem, suffix=suffix)

                    if self._skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
                        paths.append(str(out_path.relative_to(workspace.root)))
                        continue

                    voice = await self._voice_picker.pick(speaker, self._tts)
                    opts = SynthesizeOptions(
                        voice=voice,
                        rate=self._rate,
                        format=self._format,
                    )
                    try:
                        audio = await self._tts.synthesize(piece, opts)
                    except Exception as exc:
                        logger.error("TTS synth failed rec=%s idx=%d: %r", rec_id, idx, exc)
                        paths.append("")
                        continue

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(audio)
                    paths.append(str(out_path.relative_to(workspace.root)))

                new_extra = dict(rec.extra)
                tts_map = dict(new_extra.get("tts") or {})
                tts_map[target] = paths
                new_extra["tts"] = tts_map
                new_rec = replace(rec, extra=new_extra)

                if isinstance(rec_id, int):
                    await session.record_extra(rec_id, f"tts.{target}", paths)

                yield new_rec
        finally:
            session.record_fingerprint(self.name, fp)
            if owned_session:
                await asyncio.shield(session.flush(store))
            await asyncio.shield(self.aclose())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pieces_and_speakers(
        self,
        rec: SentenceRecord,
        target: str,
        translation: str,
    ) -> tuple[list[str], list[str | None]]:
        n = len(rec.segments)
        align_pieces = rec.alignment.get(target) if rec.alignment else None
        if isinstance(align_pieces, list) and len(align_pieces) == n and n > 0:
            pieces = [str(p or "") for p in align_pieces]
            speakers = [seg.speaker for seg in rec.segments]
        else:
            pieces = [translation]
            speakers = [rec.segments[0].speaker if rec.segments else None]
        return pieces, speakers

    @staticmethod
    def _stem_for(video: str, rec_id: Any, seg_idx: int) -> str:
        rid = rec_id if isinstance(rec_id, int) else -1
        return f"{video}_{rid:06d}_{seg_idx:02d}"


__all__ = ["TTSProcessor"]
