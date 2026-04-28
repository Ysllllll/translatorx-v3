"""检查器子系统的核心类型。

定义严重性级别、单条问题和聚合检查报告。
所有类型均为 frozen（不可变）。

同时导出 Scene 重构的基础类型：

- :class:`CheckContext` — 通用载荷（source/target/langs/usage/metadata）
- :class:`RuleSpec` — 规则引用（name + severity + params）
- :class:`ResolvedScene` — extends/disable/overrides 解析完毕后的不可变 scene
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from domain.model.usage import Usage


class Severity(str, Enum):
    """检查结果的严重性级别。

    继承自 :class:`str`，使枚举值可以自然地序列化为 JSON / YAML，
    并与其字符串形式相等（``Severity.ERROR == "error"``）。
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


_EMPTY_DETAILS: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Issue:
    """规则发现的单个问题。"""

    rule: str
    severity: Severity
    message: str
    details: Mapping[str, Any] = _EMPTY_DETAILS


@dataclass(frozen=True)
class CheckReport:
    """对一组翻译对运行所有规则的聚合结果。"""

    issues: tuple[Issue, ...] = ()

    @property
    def passed(self) -> bool:
        """当不存在 ERROR 级别的问题时返回 True。"""
        return not any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def failed(self) -> bool:
        """``not self.passed`` — 用于更易读的条件判断。"""
        return not self.passed

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @staticmethod
    def ok() -> CheckReport:
        return CheckReport()


# ---------------------------------------------------------------------------
# Scene 重构基础类型（P1 阶段新增）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckContext:
    """传递给每个检查/清洗步骤的通用载荷。

    翻译 scene 会填充 ``source`` / ``target`` / 语言字段。
    其他 scene（字幕、LLM 响应、术语）可能只使用 ``target`` 和 ``metadata``。

    此数据类是 **frozen** 的；调用者若要在清洗步骤后推进 ``target``，
    应使用 :func:`dataclasses.replace`。
    """

    source: str = ""
    target: str = ""
    source_lang: str = ""
    target_lang: str = ""
    usage: Usage | None = None
    prior: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """对已注册规则的引用，支持可选的覆盖参数。

    由 scene 解析（:mod:`application.checker._resolve`）产生。
    传递给规则函数，使每条规则可以统一地读取自己的参数和严重性。
    """

    name: str
    severity: Severity = Severity.ERROR
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedScene:
    """extends / disable / overrides 全部展开后的不可变 scene。

    在配置加载时由 :func:`resolve_scene` 构建一次；纯数据。
    """

    name: str
    sanitize: tuple[RuleSpec, ...] = ()
    rules: tuple[RuleSpec, ...] = ()
