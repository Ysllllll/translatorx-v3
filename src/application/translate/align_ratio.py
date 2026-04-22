"""Cross-language balance ratio for alignment validation.

Legacy ``check_ratio``: given two source-language pieces and two target-language
pieces (both expected to align pairwise), compute how "imbalanced" the split
is. A well-balanced split should yield ratio ≈ 1; the further from 1, the
worse the alignment.
"""

from __future__ import annotations

from domain.lang import LangOps


def cross_ratio(
    src_texts: list[str],
    tgt_texts: list[str],
    src_ops: LangOps,
    tgt_ops: LangOps,
) -> float:
    """Cross-length-product ratio. Always ≥ 1 (flipped if < 1).

    Formula: ``(tgt_len[0] * src_len[1]) / (tgt_len[1] * src_len[0])``.
    If either denominator is zero, returns ``float("inf")``.
    """
    if len(src_texts) != 2 or len(tgt_texts) != 2:
        raise ValueError("cross_ratio expects exactly 2 source and 2 target pieces")
    k1 = tgt_ops.length(tgt_texts[0])
    k2 = tgt_ops.length(tgt_texts[1])
    v1 = src_ops.length(src_texts[0])
    v2 = src_ops.length(src_texts[1])
    num = k1 * v2 + 1e-4
    denom = v1 * k2 + 1e-7
    if denom <= 0:
        return float("inf")
    ratio = num / denom
    if ratio < 1:
        ratio = 1.0 / ratio if ratio > 0 else float("inf")
    return ratio


__all__ = ["cross_ratio"]
