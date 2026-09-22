"""Generate backend API references from source without importing ML runtimes."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
BACKENDS = ("torch", "nnx", "linen", "tf", "keras", "mlx")


def source(module: str) -> Path:
    path = SOURCE.joinpath(*module.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def exports(module: str) -> list[str]:
    """Evaluate literal export lists and their concatenations only."""
    values = {}

    def evaluate(node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ast.literal_eval(item) for item in node.elts]
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        raise ValueError("Export lists must be statically discoverable")

    for node in ast.parse(source(module).read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = evaluate(node.value)
            except (ValueError, KeyError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values["__all__"]


def resolve(module: str, name: str, seen=None):
    """Follow local re-exports and aliases to the owning source definition."""
    seen = set() if seen is None else seen
    if (module, name) in seen:
        raise ValueError(f"Circular export: {module}.{name}")
    seen.add((module, name))
    path = source(module)
    nodes = ast.parse(path.read_text()).body
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return path, node
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            return resolve(module, node.value.id, seen)
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != name:
                    continue
                if node.level:
                    parent = (
                        module.split(".")
                        if path.name == "__init__.py"
                        else module.split(".")[:-1]
                    )
                    parent = parent[: len(parent) - node.level + 1]
                    target = ".".join(parent + ([node.module] if node.module else []))
                else:
                    target = node.module
                if target.startswith("nmn"):
                    return resolve(target, alias.name, seen)
    for node in nodes:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return path, node
    raise ValueError(f"Unresolved public export: {module}.{name}")


def signature(node, public_name: str) -> str:
    if isinstance(node, ast.Assign):
        return public_name + " = " + ast.unparse(node.value)
    if isinstance(node, ast.FunctionDef):
        return (
            public_name
            + "("
            + ast.unparse(node.args)
            + ")"
            + (" -> " + ast.unparse(node.returns) if node.returns else "")
        )
    init = next(
        (
            n
            for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ),
        None,
    )
    if init:
        args = ast.unparse(init.args)
        return (
            public_name + "(" + args.removeprefix("self, ").removeprefix("self") + ")"
        )
    fields = [
        ast.unparse(n).split(" = ")[0]
        for n in node.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and not n.target.id.startswith("_")
    ]
    return public_name + "(" + ", ".join(fields) + ")"


def generate(backend: str) -> str:
    lines = [
        f"# {backend} API reference",
        "",
        "Generated from the public export lists and source signatures. Do not edit by hand.",
        "",
        "Tensor annotations describe framework-native values; shapes, tracing, and dynamic",
        "serialization settings remain runtime contracts. See [typing support](../typing.md).",
        "",
    ]
    for name in exports("nmn." + backend):
        path, node = resolve("nmn." + backend, name)
        doc = (
            ast.get_docstring(node)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            else None
        )
        if not doc and isinstance(node, ast.ClassDef):
            doc = next(
                (
                    n.value.value
                    for n in node.body
                    if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)
                ),
                None,
            )
        lines += [
            f"## {name}",
            "",
            "```python",
            signature(node, name),
            "```",
            "",
            (doc or f"{name} is defined in `{path.relative_to(ROOT)}`.").split("\n\n")[
                0
            ],
            "",
            f"[Source](https://github.com/azettaai/nmn/blob/master/{path.relative_to(ROOT)}#L{node.lineno})",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for backend in BACKENDS:
        content = generate(backend)
        for directory in ("docs/api", "website/docusaurus/docs/api"):
            path = ROOT / directory / (backend + ".md")
            if args.check:
                if not path.exists() or path.read_text() != content:
                    raise SystemExit(
                        f"Stale API documentation: {path}; run scripts/generate_api_docs.py"
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)


if __name__ == "__main__":
    main()
