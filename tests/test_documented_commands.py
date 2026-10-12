"""The commands the documentation tells a user to type must be ones Pantheon accepts.

`--suite diagnostics` stood in README.md and docs/memory_diagnostics.md for a
release and exited 2 with "unrecognized arguments": the flag is `--test`.
Every long option on a documented `pantheon.py` or `pantheon` command line is
checked against the options of the real argument parser.
"""
import re
import shlex
from pathlib import Path

import pantheon

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
SHELL_OPERATORS = {"|", "||", "&&", ";", ">", ">>", "2>&1", "&"}


def fenced_lines(text):
    """Lines inside ``` fences, with trailing-backslash continuations joined."""
    inside, pending, out = False, "", []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside, pending = not inside, ""
            continue
        if not inside:
            continue
        line = pending + line.strip()
        if line.endswith("\\"):
            pending = line[:-1] + " "
            continue
        pending = ""
        out.append(line)
    return out


def pantheon_commands(text):
    """Argument lists of the pantheon invocations in a document's code blocks."""
    commands = []
    for line in fenced_lines(text):
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words = words[1:]          # VAR=value prefixes
        if words[:1] in (["python3"], ["python"]):
            words = words[1:]
        if not words or Path(words[0]).name not in ("pantheon.py", "pantheon"):
            continue
        args = []
        for word in words[1:]:
            if word in SHELL_OPERATORS:
                break
            args.append(word)
        commands.append(args)
    return commands


def long_options(args):
    return [word.split("=", 1)[0] for word in args if word.startswith("--") and len(word) > 2]


def parser_options():
    parser = pantheon.build_arg_parser()
    return {opt for opt in parser._option_string_actions if opt.startswith("--")}


def unknown_options(text):
    known = parser_options()
    return sorted({opt for args in pantheon_commands(text) for opt in long_options(args)} - known)


def test_every_documented_pantheon_option_exists():
    problems = {}
    seen = 0
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        seen += len(pantheon_commands(text))
        bad = unknown_options(text)
        if bad:
            problems[document.name] = bad
    assert problems == {}
    # An extractor that finds nothing would pass the check above for any text.
    assert seen >= 10


def test_the_check_catches_a_flag_that_does_not_exist():
    wrong = "```bash\npython3 pantheon.py --suite diagnostics --duration 300 --gpu 0\n```\n"
    assert unknown_options(wrong) == ["--suite"]
    right = "```bash\nCUDA_VISIBLE_DEVICES=2 python3 pantheon.py --test diagnostics --duration 300 \\\n  --gpu 0 | tee run.log\n```\n"
    assert pantheon_commands(right) == [["--test", "diagnostics", "--duration", "300", "--gpu", "0"]]
    assert unknown_options(right) == []
    assert unknown_options("```\npantheon --build-only --platform cuda\n```\n") == []
    assert unknown_options("Prose that says pantheon.py --suite is outside a code block.\n") == []


def test_every_documented_test_name_resolves():
    names = set()
    for document in DOCUMENTS:
        for args in pantheon_commands(document.read_text(encoding="utf-8")):
            if "--test" in args and args.index("--test") + 1 < len(args):
                names.add(args[args.index("--test") + 1])
    assert names
    for name in names:
        assert pantheon.resolve_test_queue(name), name
