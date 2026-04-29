"""Checker ↔ YAML 序列化与磁盘加载。

设计目标：

* **完全展开**：导出的 YAML 不使用 ``extends`` / ``disable`` / ``overrides``，
  所有 sanitize 步骤、规则、severity 与 params 全部明文列出。用户可以
  直接打开文件编辑任何参数，不需要追踪到内置预设。
* **可往返**：导出的 YAML 经 :func:`load_checker_from_yaml` 重新解析后
  得到的 :class:`Checker` 与原 Checker 在 ``check()`` 行为上等价。
* **热重载**：:meth:`Checker.reload_from_yaml` 允许长生命周期的进程
  在不重建对象的前提下，把磁盘上修改过的 YAML 重新载入到现有 Checker
  实例（并清空已编译缓存）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .scene import CheckerConfig, resolve_scene
from .types import ResolvedScene, RuleSpec


# ---------------------------------------------------------------------------
# 导出（dump）
# ---------------------------------------------------------------------------


def _yamlify(value: Any) -> Any:
    """把 tuple / Mapping 递归转成 list / dict，便于 PyYAML 输出干净的纯文本。"""
    if isinstance(value, tuple):
        return [_yamlify(v) for v in value]
    if isinstance(value, list):
        return [_yamlify(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _yamlify(v) for k, v in value.items()}
    return value


def _spec_to_dict(spec: RuleSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"name": spec.name, "severity": spec.severity.value}
    if spec.params:
        out["params"] = _yamlify(dict(spec.params))
    return out


def resolved_scene_to_payload(name: str, resolved: ResolvedScene) -> dict[str, Any]:
    """把 :class:`ResolvedScene` 渲染成可直接 ``yaml.safe_dump`` 的 dict。"""
    return {
        "checker": {
            "default_scene": name,
            "scenes": {
                name: {
                    "sanitize": [_spec_to_dict(s) for s in resolved.sanitize],
                    "rules": [_spec_to_dict(r) for r in resolved.rules],
                },
            },
        }
    }


def dump_checker_to_yaml(checker: Any, *, scene: str | None = None) -> str:
    """把 *checker* 默认 scene（或指定 scene）展开导出为完全内联的 YAML 字符串。

    Args:
        checker: :class:`Checker` 实例。
        scene:   要导出的 scene 名；默认导出 ``checker.default_scene``。

    Returns:
        YAML 文本。可写入磁盘，亦可通过 :func:`load_checker_from_yaml`
        重新载入。
    """
    name = scene or checker.default_scene
    if not name:
        raise ValueError("dump_checker_to_yaml requires a scene name (no default_scene set)")
    resolved = resolve_scene(name, checker.scenes)
    payload = resolved_scene_to_payload(name, resolved)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=200)


def write_checker_yaml(checker: Any, path: str | Path, *, scene: str | None = None) -> Path:
    """把展开的 YAML 写入 *path*，必要时创建父目录。返回最终的 :class:`Path`。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_checker_to_yaml(checker, scene=scene), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 导入（load）
# ---------------------------------------------------------------------------


def load_checker_config(path: str | Path) -> CheckerConfig:
    """从 YAML 文件解析出 :class:`CheckerConfig`。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return CheckerConfig.from_dict(data.get("checker") or data)


def load_checker_from_yaml(
    path: str | Path,
    *,
    source_lang: str = "",
    target_lang: str = "",
) -> Any:
    """从 YAML 文件构建一个全新的 :class:`Checker`。

    Args:
        path:        YAML 配置文件路径。
        source_lang: 绑定到 Checker 的源语言（用于 ``check(source, target)``）。
        target_lang: 绑定到 Checker 的目标语言。
    """
    # 推迟导入避免循环依赖。
    from .checkers import Checker

    cfg = load_checker_config(path)
    return Checker.from_config(cfg, source_lang=source_lang, target_lang=target_lang)


__all__ = [
    "dump_checker_to_yaml",
    "write_checker_yaml",
    "resolved_scene_to_payload",
    "load_checker_config",
    "load_checker_from_yaml",
]
