"""Public signature, generated API, and local link contracts (#154, #155)."""

import ast
import importlib.util
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "api_docs", ROOT / "scripts/generate_api_docs.py"
)
api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api)


def test_public_exports_have_typed_signatures_and_docstrings():
    for backend in api.BACKENDS:
        for name in api.exports("nmn." + backend):
            path, node = api.resolve("nmn." + backend, name)
            if isinstance(node, ast.Assign):
                # Framework initializer factories have inferred callable types.
                continue
            assert ast.get_docstring(node) or (
                isinstance(node, ast.ClassDef)
                and any(
                    isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)
                    for n in node.body
                )
            ), (path, name, "missing docstring")
            functions = (
                [node]
                if isinstance(node, ast.FunctionDef)
                else [
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef)
                    and (not n.name.startswith("_") or n.name == "__init__")
                ]
            )
            for function in functions:
                assert function.returns is not None, (path, name, function.name)
                for arg in (
                    function.args.posonlyargs
                    + function.args.args
                    + function.args.kwonlyargs
                ):
                    if arg.arg not in ("self", "cls"):
                        assert arg.annotation is not None, (
                            path,
                            name,
                            function.name,
                            arg.arg,
                        )


def test_generated_api_is_current():
    for backend in api.BACKENDS:
        assert (ROOT / "docs/api" / (backend + ".md")).read_text() == api.generate(
            backend
        )


def test_repository_documentation_links_resolve():
    for path in [*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")]:
        prose = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", prose):
            url = urlsplit(target.strip("<>"))
            if url.scheme or not url.path or url.path.startswith("/"):
                continue
            assert (path.parent / unquote(url.path)).exists(), (path, target)
