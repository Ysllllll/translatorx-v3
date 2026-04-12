"""Character classification utilities for multilingual text processing.
提供多语言文本处理的字符分类工具，主要用于判断中日韩(CJK)字符及处理标点符号。
"""

from __future__ import annotations


def is_cjk_ideograph(ch: str) -> bool:
    """Check if the character is a CJK Unified Ideograph.
    判断字符是否为中日韩统一表意文字（汉字）。
    涵盖了 Unicode 中的基本汉字、扩展区 A-I 以及兼容汉字。
    """
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF      # CJK Unified Ideographs (基本汉字)
        or 0x3400 <= cp <= 0x4DBF   # CJK Unified Ideographs Extension A (扩展A)
        or 0x20000 <= cp <= 0x2A6DF # CJK Unified Ideographs Extension B, C, D, E, F (扩展B-F)
        or 0x2A700 <= cp <= 0x2B73F # CJK Unified Ideographs Extension G (扩展G)
        or 0x2B740 <= cp <= 0x2B81F # CJK Unified Ideographs Extension H (扩展H)
        or 0x2B820 <= cp <= 0x2CEAF # CJK Unified Ideographs Extension I (扩展I)
        or 0xF900 <= cp <= 0xFAFF   # CJK Compatibility Ideographs (兼容汉字)
        or 0x2F800 <= cp <= 0x2FA1F # CJK Compatibility Ideographs Supplement (兼容汉字补充)
    )


def is_hangul(ch: str) -> bool:
    """Check if the character is a Korean Hangul character.
    判断字符是否为韩文字符。
    涵盖了韩文音节、字母(Jamo)以及兼容和扩展字母区。
    """
    cp = ord(ch)
    return (
        0xAC00 <= cp <= 0xD7AF      # Hangul Syllables (韩文音节)
        or 0x1100 <= cp <= 0x11FF   # Hangul Jamo (韩文字母)
        or 0x3130 <= cp <= 0x318F   # Hangul Compatibility Jamo (兼容字母)
        or 0xA960 <= cp <= 0xA97F   # Hangul Jamo Extended-A (扩展A)
        or 0xD7B0 <= cp <= 0xD7FF   # Hangul Jamo Extended-B (扩展B)
    )


def is_hiragana(ch: str) -> bool:
    """Check if the character is a Japanese Hiragana character.
    判断字符是否为日文平假名。
    """
    cp = ord(ch)
    return 0x3040 <= cp <= 0x309F   # Hiragana (平假名)


def is_katakana(ch: str) -> bool:
    """Check if the character is a Japanese Katakana character.
    判断字符是否为日文片假名及语音扩展字符。
    """
    cp = ord(ch)
    return 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF


def is_east_asian(ch: str) -> bool:
    """Check if the character is an East Asian character (Chinese, Japanese, or Korean).
    判断是否为东亚字符（即汉字、韩文、平假名或片假名的总称）。
    """
    return is_cjk_ideograph(ch) or is_hangul(ch) or is_hiragana(ch) or is_katakana(ch)


# Punctuation sets for CJK attachment
# 尾部标点（通常附着在前面词汇的末尾，如逗号、句号）
TRAILING_PUNCT = frozenset(",.!?:;，。！？：；、")
# 闭合标点（通常附着在前面词汇的末尾，如右括号、右引号）
CLOSING_PUNCT = frozenset(")]}）》”’")  # ）》”’
# 起始标点（通常附着在后面词汇的开头，如左括号、左引号）
OPENING_PUNCT = frozenset("([{（《“‘")   # （《“‘

# 需要依附于前一个 Token 的标点集合
ATTACH_TO_PREV = TRAILING_PUNCT | CLOSING_PUNCT

# Comprehensive punctuation for strip_punc operations
# 用于去除词汇两端标点的完整标点符号集合（合并尾部、闭合、起始及其他特殊标点）
STRIP_PUNCT = "".join(sorted(
    TRAILING_PUNCT | CLOSING_PUNCT | OPENING_PUNCT
    | set("¡¿<>\"'—–‐…·「」『』【】")
))


def is_opening_punct_char(ch: str) -> bool:
    """Check if the character is an opening punctuation.
    判断是否为起始标点（如左括号、左引号）。
    """
    return ch in OPENING_PUNCT


def is_attach_to_prev_char(ch: str) -> bool:
    """Check if the character should attach to the previous token.
    判断标点是否应该依附于前一个词（即尾部标点或闭合标点）。
    """
    return ch in ATTACH_TO_PREV


def cjk_needs_space(prev_last: str, curr_first: str) -> bool:
    """Determine if a space is needed between two characters in CJK context.
    判断在 CJK 文本处理中，两个相邻字符之间是否需要插入空格。
    如果两个字符都是东亚字符，则不需要空格；否则（如中英文混合），通常需要空格。
    """
    # 如果任一字符不是字母/数字（例如是标点符号），由标点规则处理，不强制加空格
    if not prev_last.isalnum() or not curr_first.isalnum():
        return False
    # 当且仅当两个字符不全为东亚字符时（即存在拉丁字母与中日韩交替，或全拉丁），才需要空格
    return not (is_east_asian(prev_last) and is_east_asian(curr_first))


# Characters treated as content (not punctuation) in CJK mode
# 在 CJK 模式下被视为内容（而非标点）的特殊字符，例如省略号和间隔号
CONTENT_LIKE_CHARS = {"…", "・"}


def decompose_token(token: str) -> tuple[str, str, str]:
    """Decompose a token into (leading_punct, content, trailing_punct).
    将一个 Token 拆分为三部分：前导标点、核心内容、尾部标点。
    主要用于在分词后准确提取出词汇的核心内容。
    
    Uses STRIP_PUNCT to identify punctuation characters.
    """
    i = 0
    while i < len(token) and token[i] in STRIP_PUNCT:
        i += 1
    j = len(token)
    while j > i and token[j - 1] in STRIP_PUNCT:
        j -= 1
    return (token[:i], token[i:j], token[j:])
