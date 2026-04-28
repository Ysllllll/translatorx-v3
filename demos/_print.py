"""Lightweight printing helpers shared by all demos.

Goals:

* Zero heavy deps — no llm / adapters / engine factories. Importable from
  any plain demo without dragging in the full ``_shared`` machinery.
* Single visual style — every demo uses ``section`` / ``step`` / ``kv`` /
  ``info`` / ``ok`` / ``warn`` / ``err`` so the output across the whole
  ``demos/`` tree looks the same.
* Optional rich — falls back to plain ``print`` when ``rich`` is not
  installed so demos still work in minimal environments.

Usage::

    from _print import section, step, kv, info, ok, warn, err

    section("Phase 1", "Lang ops factory")
    step("1.1", "tokenize", "expect 7 tokens")
    info("language detected: en")
    kv("tokens", 7)
    ok("done")
"""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table

    _console: Console | None = Console()
    _HAVE_RICH = True
except Exception:  # pragma: no cover - rich missing
    _console = None
    _HAVE_RICH = False


def console() -> Any:
    """Return the shared rich Console (or ``None`` if rich missing)."""
    return _console


def section(label: str, title: str) -> None:
    """Major heading. Use once per top-level scenario in a demo."""
    if _HAVE_RICH and _console is not None:
        _console.print()
        _console.print(Rule(f"[bold cyan]{label}[/bold cyan] — [bold]{title}[/bold]", style="cyan"))
    else:
        print()
        print(f"=== {label} — {title} ===")


def step(label: str, title: str, expected: str = "") -> None:
    """Numbered step under a section. ``expected`` is a one-line note."""
    if _HAVE_RICH and _console is not None:
        _console.print()
        _console.print(
            Rule(
                f"[bold green]{label}[/bold green] — [bold]{title}[/bold]",
                style="green",
            )
        )
        if expected:
            _console.print(f"[dim]{expected}[/dim]")
    else:
        print()
        print(f"--- {label} — {title} ---")
        if expected:
            print(f"  ({expected})")


def info(msg: str) -> None:
    if _HAVE_RICH and _console is not None:
        _console.print(f"[dim]{msg}[/dim]")
    else:
        print(msg)


def ok(msg: str) -> None:
    if _HAVE_RICH and _console is not None:
        _console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"[ok] {msg}")


def warn(msg: str) -> None:
    if _HAVE_RICH and _console is not None:
        _console.print(f"[yellow]![/yellow] {msg}")
    else:
        print(f"[warn] {msg}")


def err(msg: str) -> None:
    if _HAVE_RICH and _console is not None:
        _console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"[err] {msg}")


def kv(key: str, value: Any, *, width: int = 20) -> None:
    """Render a key/value line for compact diagnostic output."""
    key_str = f"{key:<{width}}"
    if _HAVE_RICH and _console is not None:
        _console.print(f"  [bold]{key_str}[/bold]  {value}")
    else:
        print(f"  {key_str}  {value}")


def table(title: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Render a compact table from dict rows."""
    if _HAVE_RICH and _console is not None:
        tbl = Table(title=title, show_lines=False)
        for col in columns:
            tbl.add_column(col)
        for row in rows:
            tbl.add_row(*(str(row.get(col, "")) for col in columns))
        _console.print(tbl)
        return

    print(title)
    if not rows:
        print("  (empty)")
        return
    widths = {col: max(len(str(col)), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
    print(header)
    print("-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        print(" | ".join(f"{str(row.get(col, '')):<{widths[col]}}" for col in columns))


def banner(text: str) -> None:
    """One-shot banner for the very top of a demo."""
    if _HAVE_RICH and _console is not None:
        _console.print()
        _console.print(Rule(f"[bold magenta]{text}[/bold magenta]", style="magenta"))
        _console.print()
    else:
        print()
        print(f"##### {text} #####")
        print()


__all__ = [
    "banner",
    "console",
    "err",
    "info",
    "kv",
    "ok",
    "section",
    "step",
    "table",
    "warn",
]
