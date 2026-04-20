"""EN-type ops for space-delimited languages (en, ru, es, fr, de, pt, vi)."""

from __future__ import annotations

import re

from ._core._base_ops import _BaseOps, normalize_mode, _VALID_MODES


# ---------------------------------------------------------------------------
# Per-language abbreviation sets.
# Each entry is the word that appears *before* a period.
# Comments use English / 中文 for cross-language readability.
# ---------------------------------------------------------------------------

_ABBREVIATIONS: dict[str, frozenset[str]] = {
    # -- English 英语 --
    "en": frozenset(
        {
            # Titles 称谓
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Sr",
            "Jr",
            "St",
            # Business 商业
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            # Miscellaneous 其他
            "vs",
            "etc",
            "eg",
            "ie",
            # Months 月份
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        }
    ),
    # -- Russian 俄语 --
    "ru": frozenset(
        {
            # English base (shared in Russian formal text) 英语基础缩写
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "vs",
            "etc",
            "eg",
            "ie",
            # Currency / numbers 货币/数字
            "руб",  # рубль — ruble 卢布
            "млн",  # миллион — million 百万
            "млрд",  # миллиард — billion 十亿
            "тыс",  # тысяча — thousand 千
            # Titles / places 称谓/地名
            "г",  # год/город — year/city 年/城市
            "ул",  # улица — street 街道
            "пр",  # проспект — avenue 大街
            # Academic 学术
            "напр",  # например — for example 例如
            "прибл",  # приблизительно — approximately 大约
            "см",  # смотри — see 参见
            "ок",  # около — approximately 约
            "тел",  # телефон — phone 电话
            "зам",  # заместитель — deputy 副职
            "зав",  # заведующий — head 负责人
        }
    ),
    # -- French 法语 --
    "fr": frozenset(
        {
            # Titles 称谓
            "Mme",  # Madame — Mrs. 女士(已婚)
            "Mlle",  # Mademoiselle — Miss 小姐
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Sr",
            "Jr",
            # Places 地名
            "av",  # avenue — avenue 大街
            "bd",  # boulevard — boulevard 林荫大道
            "qu",  # quai — quay 码头/堤岸
            # Academic 学术
            "éd",  # édition — edition 版本
            "réf",  # référence — reference 参考
            "env",  # environ — approximately 大约
            "vol",  # volume — volume 卷/册
            "fig",  # figure — figure 图
            "no",  # numéro — number 编号
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "vs",
            "etc",
            "eg",
            "ie",
            # Months 月份
            "janv",  # janvier — January 一月
            "févr",  # février — February 二月
            "avr",  # avril — April 四月
            "juil",  # juillet — July 七月
            "sept",  # septembre — September 九月
            "oct",  # octobre — October 十月
            "nov",  # novembre — November 十一月
            "déc",  # décembre — December 十二月
        }
    ),
    # -- German 德语 --
    "de": frozenset(
        {
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Sr",
            "Jr",
            "St",
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "vs",
            "etc",
            "eg",
            "ie",
            # Common abbreviations 常用缩写
            "usw",  # und so weiter — and so on 等等
            "bzw",  # beziehungsweise — respectively 即/分别是
            "evtl",  # eventuell — possibly 可能
            "ca",  # circa — approximately 大约
            "Hr",  # Herr — Mr. 先生
            "Fr",  # Frau — Mrs./Ms. 女士
            "Hrsg",  # Herausgeber — editor 编辑
            "Aufl",  # Auflage — edition 版次
            "Jh",  # Jahrhundert — century 世纪
            # Months 月份
            "Jan",
            "Feb",
            "Mär",
            "Apr",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Okt",
            "Nov",
            "Dez",
        }
    ),
    # -- Spanish 西班牙语 --
    "es": frozenset(
        {
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Sr",
            "Jr",
            "St",
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "vs",
            "etc",
            "eg",
            "ie",
            # Titles 称谓
            "Ud",  # Usted — you (formal) 您(尊称)
            "Sra",  # Señora — Mrs. 女士(已婚)
            "Srta",  # Señorita — Miss 小姐
            "Dra",  # Doctora — Dr. (female) 女博士
            "Profa",  # Profesora — Prof. (female) 女教授
            "Lic",  # Licenciado/a — graduate 学士
            # Places 学术/地名
            "av",  # avenida — avenue 大街
            "pág",  # página — page 页
            "tel",  # teléfono — phone 电话
            "aprox",  # aproximadamente — approximately 大约
            # Months 月份
            "ene",
            "feb",
            "abr",
            "may",
            "jun",
            "jul",
            "ago",
            "sep",
            "oct",
            "nov",
            "dic",
        }
    ),
    # -- Portuguese 葡萄牙语 --
    "pt": frozenset(
        {
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Sr",
            "Jr",
            "St",
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "vs",
            "etc",
            "eg",
            "ie",
            # Titles 称谓
            "Sra",  # Senhora — Mrs. 女士(已婚)
            "Dra",  # Doutora — Dr. (female) 女博士
            "Profa",  # Professora — Prof. (female) 女教授
            # Places 学术/地名
            "av",  # avenida — avenue 大街
            "pág",  # página — page 页
            "tel",  # telefone — phone 电话
            "aprox",  # aproximadamente — approximately 大约
            # Months 月份
            "jan",
            "fev",
            "mar",
            "abr",
            "mai",
            "jun",
            "jul",
            "ago",
            "set",
            "out",
            "nov",
            "dez",
        }
    ),
    # -- Vietnamese 越南语 --
    "vi": frozenset(
        {
            "Mr",
            "Mrs",
            "Ms",
            "Dr",
            "Prof",
            "Sr",
            "Jr",
            "St",
            "Inc",
            "Ltd",
            "Co",
            "Corp",
            "vs",
            "etc",
            "eg",
            "ie",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            # Vietnamese academic/organizational 越南语学术/机构
            "GS",  # Giáo sư — Professor 教授
            "TS",  # Tiến sĩ — PhD 博士
            "ThS",  # Thạc sĩ — Master's 硕士
            "KS",  # Kỹ sư — Engineer 工程师
            "TP",  # Thành phố — city 城市
            "ĐC",  # Địa chỉ — address 地址
            "ĐT",  # Điện thoại — phone 电话
            "VN",  # Việt Nam — Vietnam 越南
            "khoảng",  # approximately 大约
        }
    ),
}


class EnTypeOps(_BaseOps):
    def __init__(self, language: str = "en") -> None:
        self._language = language

    @property
    def sentence_terminators(self) -> frozenset[str]:
        return frozenset({".", "!", "?"})

    @property
    def clause_separators(self) -> frozenset[str]:
        return frozenset({",", ";", ":", "—"})

    @property
    def abbreviations(self) -> frozenset[str]:
        return _ABBREVIATIONS.get(self._language, _ABBREVIATIONS["en"])

    @property
    def is_cjk(self) -> bool:
        return False

    def split(self, text: str, mode: str = "word", attach_punctuation: bool = True) -> list[str]:
        mode = normalize_mode(mode)
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {mode!r}")
        if mode == "character":
            return [ch for ch in text if not ch.isspace()]
        if attach_punctuation and self._language != "fr":
            return self.normalize(text).split()
        return text.split()

    def join(self, tokens: list[str]) -> str:
        if not tokens:
            return ""

        parts: list[tuple[bool, str]] = []  # (space_before, text)
        open_double = False
        skip_next = False

        for i, token in enumerate(tokens):
            is_first = i == 0
            is_single = len(token) == 1

            space = not is_first

            if skip_next:
                space = False
                skip_next = False

            if is_single and token == '"':
                if open_double:
                    space = False
                    if parts and parts[-1][0] and len(parts[-1][1]) == 1 and not parts[-1][1].isalnum():
                        parts[-1] = (False, parts[-1][1])
                    open_double = False
                else:
                    skip_next = True
                    open_double = True
            elif is_single and token == "'":
                space = False
                skip_next = True
            elif is_single and token in (",", "."):
                space = False
            elif is_single and token in (")", "]", "}"):
                space = False
            elif is_single and token in ("(", "[", "{", "¡", "¿"):
                skip_next = True

            parts.append((space, token))

        result: list[str] = []
        for sp, text in parts:
            if sp:
                result.append(" ")
            result.append(text)
        return "".join(result)

    def length(self, text: str, **kwargs: int) -> int:
        return len(text)

    def normalize(self, text: str) -> str:
        if self._language == "fr":
            text = re.sub(r" +([.,])", r"\1", text)
            text = re.sub(r" +([!?;:])", r" \1", text)
            text = re.sub(r"([^\s!?;:,.])([!?;])", r"\1 \2", text)
            text = re.sub(r"([a-zA-Z\u00C0-\u024F])(:)", r"\1 \2", text)
        else:
            text = re.sub(r" +([.,!?);:}\]])", r"\1", text)
            text = re.sub(r"([(\[{¡¿]) +", r"\1", text)
        return text
