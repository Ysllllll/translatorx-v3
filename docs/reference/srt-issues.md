# Subtitle Quality Issues Catalog

> Analyzed from 13,309 SRT files and 11,324 WhisperX JSON files (course
> lecture subtitles).
> Last updated: 2025-04-15

This document catalogs all quality issues discovered in real-world subtitle
data. Each issue includes its frequency, impact on the subtitle pipeline, and
the recommended fix.

- **SRT issues** → preprocessed by `subtitle.io.srt.sanitize_srt`
- **WhisperX issues** → preprocessed by `subtitle.io.whisperx.sanitize_whisperx`

---

## Issue Classification

Issues are grouped into four categories:

1. **File-level format** — encoding, line endings, BOM
2. **Timestamp** — overlaps, zero/micro duration, non-standard format
3. **Text content** — invisible chars, tags, bracket annotations, punctuation
4. **Structural** — speaker labels, dialogue markers, empty/duplicate segments

---

## 1. File-Level Format Issues

### 1.1 CRLF Line Endings

| Metric | Value |
|--------|-------|
| Count  | 239 files |
| Impact | May cause parsing issues if `\r` leaks into text |
| Fix    | Normalize to `\n` before parsing |

### 1.2 BOM (Byte Order Mark)

| Metric | Value |
|--------|-------|
| Count  | 0 (not detected in this dataset, but common in the wild) |
| Impact | BOM appears as invisible char at start of first segment |
| Fix    | Strip `\ufeff` from start of content |

### 1.3 Encoding Errors

| Metric | Value |
|--------|-------|
| Count  | 8 files |
| Impact | Replacement character `\ufffd` appears in text |
| Fix    | Detect and warn; optionally skip affected segments |

---

## 2. Timestamp Issues

### 2.1 Timestamp Overlap

| Metric | Value |
|--------|-------|
| Count  | 1,385 occurrences across 6 files |
| Impact | Non-monotonic timestamps break alignment assumptions |
| Fix    | Clamp: `seg[i].start = max(seg[i].start, seg[i-1].end)` |

Example: A few files have massive overlaps (multi-track concatenation).

### 2.2 Zero-Duration Segments

| Metric | Value |
|--------|-------|
| Count  | 81 across 35 files |
| Impact | Segment has no time span; word timing impossible |
| Fix    | Remove segment, or merge text into adjacent segment |

### 2.3 Micro-Duration Segments (< 0.05s)

| Metric | Value |
|--------|-------|
| Count  | 998 across 127 files |
| Impact | Too short to display; often contain `[INAUDIBLE]` or annotation |
| Fix    | Merge into adjacent segment if text is meaningful |

### 2.4 Period in Timestamp Separator

| Metric | Value |
|--------|-------|
| Count  | 0 (standard format uses comma `00:00:00,000`) |
| Impact | Non-standard SRT uses `00:00:00.000`; parser rejects it |
| Fix    | Accept both `,` and `.` as millisecond separator |

### 2.5 Extra Data After Timestamp

| Metric | Value |
|--------|-------|
| Count  | 0 in this dataset (common in WebVTT-derived SRTs) |
| Impact | Positioning info like `position:50% align:center` after `-->` |
| Fix    | Strip everything after the second timestamp |

---

## 3. Text Content Issues

### 3.1 Invisible / Zero-Width Characters

| Metric | Value |
|--------|-------|
| Count  | 416 control chars (37 files), 409 zero-width chars (30 files) |
| Impact | Invisible chars break text matching and length calculation |
| Fix    | Strip all Unicode categories Cf (format chars) except standard whitespace |

Characters found: `U+2060` (Word Joiner), `U+200B` (Zero Width Space),
`U+200E`/`U+200F` (LTR/RTL marks), `U+FEFF` (BOM as zero-width no-break space).

### 3.2 HTML Formatting Tags

| Metric | Value |
|--------|-------|
| Count  | 101 across 31 files |
| Impact | Tags like `<u>`, `<b>`, `<i>`, `<font>` appear in text |
| Fix    | Strip all HTML tags, preserve inner text |

Example: `read<u>grades.</u>` → `readgrades.` (with appropriate space handling)

### 3.3 ASS/SSA Style Tags

| Metric | Value |
|--------|-------|
| Count  | 0 in this dataset |
| Impact | Tags like `{\an8}`, `{\pos(x,y)}` in text |
| Fix    | Strip `{\\...}` patterns |

### 3.4 Non-Standard Spaces

| Metric | Value |
|--------|-------|
| Count  | 2 occurrences across 2 files |
| Impact | Non-breaking space `\u00a0`, em space `\u2003`, etc. |
| Fix    | Replace with standard space `U+0020` |

### 3.5 Double Spaces

| Metric | Value |
|--------|-------|
| Count  | 2,078 across 218 files |
| Impact | Affects length calculation and word splitting |
| Fix    | Collapse consecutive spaces to single space |

### 3.6 Trailing Whitespace

| Metric | Value |
|--------|-------|
| Count  | 8,507 across 567 files |
| Impact | Minor; may affect text comparison |
| Fix    | Strip trailing whitespace from text lines |

### 3.7 Smart Quotes

| Metric | Value |
|--------|-------|
| Count  | 9,121 across 919 files |
| Impact | Inconsistent quote characters `""''` vs `"'` |
| Fix    | Normalize to ASCII quotes |

### 3.8 Unicode Ellipsis

| Metric | Value |
|--------|-------|
| Count  | 547 across 172 files |
| Impact | `…` (U+2026) vs `...` — inconsistent ellipsis representation |
| Fix    | Normalize `…` to `...` |

### 3.9 Double Period

| Metric | Value |
|--------|-------|
| Count  | 207 across 144 files |
| Impact | `..` (not `...`) causes sentence splitting to insert space → text mismatch |
| Fix    | Normalize `..` (non-ellipsis) to `.` |

Example: `etc., etc..` → `etc., etc.`

### 3.10 Leading Punctuation

| Metric | Value |
|--------|-------|
| Count  | 579 across 293 files |
| Impact | Sentence starts with `.`, `,`, etc. — confuses sentence boundary detection |
| Fix    | Move leading punctuation to end of previous segment's text |

Example: `./buggy is not going to print 3.`

---

## 4. Structural Issues

### 4.1 Bracket-Only Segments

| Metric | Value |
|--------|-------|
| Count  | 9,830 across 3,282 files |
| Impact | Non-speech content occupies timeline; gets merged into adjacent sentences |
| Fix    | Mark as non-text or remove; preserve timestamp gap |

Common values: `[MUSIC PLAYING]`, `[APPLAUSE]`, `[MUSIC]`, `[LAUGHTER]`

### 4.2 Inline Bracket Annotations

| Metric | Value |
|--------|-------|
| Count  | 13,606 across 2,466 files |
| Impact | `[INAUDIBLE]`, `[CROSSTALK]` embedded in text |
| Fix    | Remove inline brackets; leave surrounding text intact |

Example: `STUDENT: [INAUDIBLE] something` → `STUDENT: something`

### 4.3 Speaker Labels

| Metric | Value |
|--------|-------|
| Count  | 12,715 across 635 files |
| Impact | `NAME:` prefix occupies text length, affects splitting |
| Fix    | Extract label into `segment.speaker` field; remove from text |

Pattern: `^[A-Z][A-Z\s\d]*:` (e.g., `DAVID MALAN:`, `STUDENT:`, `SPEAKER 1:`)

### 4.4 Dialogue Dashes

| Metric | Value |
|--------|-------|
| Count  | 4,741 across 424 files |
| Impact | `- text` or `— text` prefix occupies space |
| Fix    | Strip leading dash; optionally infer speaker change |

### 4.5 Empty Text Segments

| Metric | Value |
|--------|-------|
| Count  | 637 across 129 files |
| Impact | Segments with no text content |
| Fix    | Remove |

### 4.6 No-Timestamp Blocks

| Metric | Value |
|--------|-------|
| Count  | 820 across 626 files |
| Impact | Malformed SRT blocks without `-->` timestamp |
| Fix    | Skip during parsing (current parser already does this) |

---

---

# Part B: WhisperX JSON Issues

> Analyzed from 11,324 WhisperX JSON files.

WhisperX produces word-level timestamps via forced alignment, but the output
contains systematic quality issues that must be fixed before pipeline
processing.

---

## 5. Word Timestamp Issues

### 5.1 Words Without Timestamps

| Metric | Value |
|--------|-------|
| Count  | 635,790 words across 10,386 files |
| Impact | **Root cause of most pipeline failures.** Words without `start`/`end` get dropped or default to 0, causing timeline gaps after `split()` |
| Fix    | Interpolate timestamps from neighboring timed words |

Breakdown of untimed word types:

| Type | Count | Example |
|------|-------|---------|
| Numbers / years | 560,562 | `1876.`, `1865,` |
| Numeric expressions | 71,152 | `86,000`, `$100`, `10%` |
| Other tokens | 4,001 | `&`, `C++`, special chars |
| Music symbols | 75 | `♪` |

WhisperX's forced alignment fails on tokens that don't match audio
phonemes — numbers spoken as words (e.g., "eighteen sixty-five" but
transcribed as `1865`), currency/percentage expressions, and music
symbols.

### 5.2 Low-Score Words

| Metric | Value |
|--------|-------|
| Count  | 533,462 across 10,281 files |
| Impact | Alignment confidence < 0.1; timestamps may be inaccurate |
| Fix    | Flag but preserve; use for quality reporting |

### 5.3 Very Long Word Duration (> 5s)

| Metric | Value |
|--------|-------|
| Count  | 6,351 across 2,993 files |
| Impact | Single word spanning > 5 seconds indicates alignment error |
| Fix    | Cap word duration; redistribute excess to adjacent words |

### 5.4 Word Overlap

| Metric | Value |
|--------|-------|
| Count  | (included in segment overlap count) |
| Impact | Word `start` < previous word `end` |
| Fix    | Clamp: `word[i].start = max(word[i].start, word[i-1].end)` |

### 5.5 Zero-Duration Words

| Metric | Value |
|--------|-------|
| Count  | (subset of timed words) |
| Impact | Word has `start == end`; no time span |
| Fix    | Assign minimum duration or merge into adjacent word |

---

## 6. Segment-Level Issues

### 6.1 Segment Start > End

| Metric | Value |
|--------|-------|
| Count  | 1,928 across 1,051 files |
| Impact | Inverted timestamps — negative duration |
| Fix    | Swap start/end, or skip segment |

### 6.2 Segment Overlap

| Metric | Value |
|--------|-------|
| Count  | 896 across 664 files |
| Impact | Non-monotonic segment timeline |
| Fix    | Clamp start to previous segment's end |

### 6.3 Leading Punctuation in Text

| Metric | Value |
|--------|-------|
| Count  | 432 across 161 files |
| Impact | Colon, comma, period at text start |
| Fix    | Same as SRT — move to previous segment or strip |

---

## 7. Word Text Issues

### 7.1 Very Long Word Text (> 30 chars)

| Metric | Value |
|--------|-------|
| Count  | 218 across 173 files |
| Impact | Single "word" contains concatenated text (WhisperX bug) |
| Fix    | Replace with `...` if all uppercase; flag otherwise |

Example: `Spanish-American-Cuban-Philippine` (hyphenated compound — valid),
vs `ENVIRONMENTALISTENVIRONMENTALIST` (bug — replace).

### 7.2 Repeating Word Patterns

| Metric | Value |
|--------|-------|
| Count  | 1,169 (2-word patterns), 594 (3-word patterns) |
| Impact | WhisperX hallucination: same words repeat 4+ times |
| Fix    | Detect and collapse to single occurrence |

Example: `iPhone 7?s iPhone 7?s iPhone 7?s iPhone 7?s` → `iPhone 7?s`

### 7.3 Consecutive Untimed Duplicates

| Metric | Value |
|--------|-------|
| Count  | 8,650 across 1,694 files |
| Impact | Same untimed word repeated adjacently |
| Fix    | Deduplicate consecutive untimed words |

Example: `♪ ♪ ♪ ♪` (all without timestamps) → `♪`

---

## Data Source

- **Location**: `/home/ysl/workspace/all_course2/**/zzz_subtitle/`
- **SRT files**: 13,309
- **JSON files**: 11,324
- **Content**: Lecture subtitles from various university courses
- **Languages**: Primarily English
- **Analysis script**: `benchmark/validate_subtitles.py`
