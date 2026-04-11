"""Language code normalization."""

from __future__ import annotations

_ALIASES: dict[str, str] = {}

_RAW_MAP: dict[str, list[str]] = {
    "zh": ["zh", "chinese", "cn", "中文", "汉语"],
    "en": ["en", "english", "英语"],
    "ru": ["ru", "russian", "русский", "俄语"],
    "es": ["es", "spanish", "español", "西班牙语"],
    "ja": ["ja", "japanese", "日本語", "日语"],
    "ko": ["ko", "korean", "한국어", "韩语"],
    "fr": ["fr", "french", "français", "法语"],
    "de": ["de", "german", "deutsch", "德语"],
    "pt": ["pt", "portuguese", "português", "葡萄牙语"],
    "vi": ["vi", "vietnamese", "tiếng việt", "越南语"],
}

for _code, _aliases in _RAW_MAP.items():
    for _alias in _aliases:
        _ALIASES[_alias.lower()] = _code


def normalize_language(value: str) -> str:
    key = value.strip().lower()
    if not key:
        raise ValueError(f"Unsupported language: {value!r}")
    code = _ALIASES.get(key)
    if code is None:
        raise ValueError(f"Unsupported language: {value!r}")
    return code
