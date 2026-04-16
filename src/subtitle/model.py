"""Backward-compatibility shim — types now live in the top-level ``model`` package."""

from model import Segment, SentenceRecord, Word

__all__ = ["Word", "Segment", "SentenceRecord"]
