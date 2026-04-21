"""Architecture guard — enforce Hexagonal layering (inward-only deps).

Layers (top=outermost)::

    api        → application, adapters, ports, domain
    application→ adapters, ports, domain
    adapters   → ports, domain
    ports      → domain
    domain     → (nothing in-project)

A violation means a lower layer imports a higher layer, or a peer that
would create a cycle.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

# Allowed deps for each layer (layer → set of allowed layers)
ALLOWED = {
    "domain": set(),
    "ports": {"domain"},
    "adapters": {"domain", "ports"},
    "application": {"domain", "ports", "adapters"},
    "api": {"domain", "ports", "adapters", "application"},
}

LAYERS = tuple(ALLOWED.keys())


def _top_layer(module: str) -> str | None:
    head = module.split(".", 1)[0]
    return head if head in LAYERS else None


def _iter_imports(tree: ast.AST) -> list[str]:
    """Runtime imports only — skips anything under `if TYPE_CHECKING:`."""
    mods: list[str] = []

    def _is_type_checking(node: ast.If) -> bool:
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        if (
            isinstance(test, ast.Attribute)
            and isinstance(test.value, ast.Name)
            and test.value.id == "typing"
            and test.attr == "TYPE_CHECKING"
        ):
            return True
        return False

    def _visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If) and _is_type_checking(node):
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mods.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mods.append(node.module)
            elif hasattr(node, "body") and isinstance(getattr(node, "body"), list):
                _visit(node.body)  # type: ignore[arg-type]
                if hasattr(node, "orelse") and isinstance(node.orelse, list):
                    _visit(node.orelse)

    _visit(tree.body)  # type: ignore[attr-defined]
    return mods


@pytest.mark.parametrize("py_file", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_upward_import(py_file: Path) -> None:
    rel = py_file.relative_to(SRC)
    parts = rel.parts
    if not parts or parts[0] not in LAYERS:
        pytest.skip(f"not under a known layer: {rel}")
    own_layer = parts[0]
    allowed = ALLOWED[own_layer] | {own_layer}

    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for mod in _iter_imports(tree):
        top = _top_layer(mod)
        if top is None:
            continue
        assert top in allowed, f"{rel} (layer={own_layer}) imports '{mod}' (layer={top}); allowed layers: {sorted(allowed)}"
