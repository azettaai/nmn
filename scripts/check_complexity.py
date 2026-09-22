"""Reject growth in per-module branch complexity relative to the audited baseline."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def complexity(text: str) -> int:
    """Count decision points, including boolean expressions and comprehensions."""
    return sum(
        len(node.values) - 1 if isinstance(node, ast.BoolOp) else 1
        for node in ast.walk(ast.parse(text))
        if isinstance(
            node,
            (
                ast.If,
                ast.IfExp,
                ast.For,
                ast.While,
                ast.ExceptHandler,
                ast.comprehension,
                ast.BoolOp,
            ),
        )
    )


def main() -> None:
    baseline = json.loads((ROOT / "scripts/complexity-baseline.json").read_text())
    errors = []
    for path in (ROOT / "src/nmn").rglob("*.py"):
        if "examples" in path.parts or path.name == "_version.py":
            continue
        relative = str(path.relative_to(ROOT))
        allowed = baseline.get(relative, 10)
        count = complexity(path.read_text())
        if count > allowed:
            errors.append(f"{relative}: {count} decision points exceeds {allowed}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
