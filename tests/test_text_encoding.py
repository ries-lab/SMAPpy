"""Every text file smappy opens names its encoding.

Without one Python takes the locale's, cp1252 on Windows: a config, chain or
workspace with a "µ" or a "⋯" in it then fails to be written there
(UnicodeEncodeError, as the tutorial player's did in 2026-10), or is written
in cp1252 and read back as mojibake on Linux and macOS.  So every
`read_text`, `write_text` and text-mode `open` in the package says
``encoding=``; binary modes need none.  The scan reads the source with `ast`
and imports nothing.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "smappy"


def _mode(call: ast.Call, position: int):
    """The mode argument, or None where it is left to the default "r"."""
    for k in call.keywords:
        if k.arg == "mode":
            return k.value
    return call.args[position] if len(call.args) > position else None


def _binary(mode) -> bool:
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def _unnamed(call: ast.Call) -> bool:
    """True for a text-mode call that leaves the encoding to the locale."""
    if any(k.arg == "encoding" for k in call.keywords):
        return False
    func = call.func
    if isinstance(func, ast.Name) and func.id == "open":           # builtin open(file, mode)
        return not _binary(_mode(call, 1))
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "fdopen":
        return not _binary(_mode(call, 1))
    if func.attr == "read_text":
        # a positional argument is importlib.metadata's file name
        # (`Distribution.read_text`, which decodes UTF-8 itself) or the encoding
        return not call.args
    if func.attr == "write_text":
        return len(call.args) < 2
    if func.attr == "open":
        # Path.open(mode); `Image.open(path)`, `wave.open(..)` and the GUI's
        # own open() take no mode string first, so only a literal mode counts
        mode = _mode(call, 0)
        return (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
                and set(mode.value) <= set("rwaxt+") and not _binary(mode))
    return False


def test_every_text_file_smappy_reads_or_writes_names_its_encoding():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        found += [f"{path.relative_to(PACKAGE).as_posix()}:{n.lineno}: {ast.unparse(n)[:80]}"
                  for n in ast.walk(tree) if isinstance(n, ast.Call) and _unnamed(n)]
    assert not found, "no encoding= (say encoding=\"utf-8\"):\n" + "\n".join(found)


def test_the_scan_catches_a_text_open_and_lets_a_binary_one_pass():
    def calls(source):
        return [_unnamed(n) for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)]
    assert calls("open(p)") == [True]
    assert calls("open(p, 'w')") == [True]
    assert calls("p.write_text(t)") == [True]
    assert calls("p.read_text()") == [True]
    assert calls("p.open('a')") == [True]
    assert calls("open(p, 'rb')") == [False]
    assert calls("open(p, mode='wb')") == [False]
    assert calls("p.read_text(encoding='utf-8')") == [False]
    assert calls("dist.read_text('INSTALLER')") == [False]
    assert calls("Image.open(p)") == [False]
