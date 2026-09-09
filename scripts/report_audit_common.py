"""Shared LaTeX handling for the report audit scripts.

`check_report_numbers.py` and `prose_stats.py` both have to turn a `.tex` section
into something they can count. They need slightly different things out of it --
one wants numeric tokens with their line numbers, the other wants sentences --
but the stripping rules are the same, so they live here.
"""

from __future__ import annotations

import re

# Commands whose braced argument is never prose and never a quantitative claim.
# Their contents are removed wholesale (keys, labels, paths, image options).
DROP_WITH_ARG = (
    "label",
    "ref",
    "eqref",
    "pageref",
    "cite",
    "citep",
    "citet",
    "citealp",
    "citeauthor",
    "citeyear",
    "input",
    "include",
    "includegraphics",
    "graphicspath",
    "bibliography",
    "bibliographystyle",
    "usepackage",
    "documentclass",
    "newcommand",
    "renewcommand",
    "hypersetup",
)

# Commands whose argument IS prose and should be kept, minus the markup.
KEEP_ARG = (
    "emph",
    "textit",
    "textbf",
    "texttt",
    "textsc",
    "underline",
    "caption",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "item",
    "footnote",
)

_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_ENV = re.compile(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?")


def strip_comments(text: str) -> str:
    """Drop `%` comments, keeping escaped `\\%`."""
    return _COMMENT.sub("", text)


def _drop_command_with_arg(text: str, name: str) -> str:
    """Remove `\\name[opt]{arg}` including nested braces in the argument."""
    out = []
    i = 0
    pattern = "\\" + name
    while i < len(text):
        j = text.find(pattern, i)
        if j < 0:
            out.append(text[i:])
            break
        # Must not be a prefix of a longer command name.
        after = j + len(pattern)
        if after < len(text) and (text[after].isalpha() or text[after] == "*"):
            out.append(text[i : j + len(pattern)])
            i = j + len(pattern)
            continue
        out.append(text[i:j])
        k = after
        while k < len(text) and text[k] == "*":
            k += 1
        if k < len(text) and text[k] == "[":
            depth = 0
            while k < len(text):
                if text[k] == "[":
                    depth += 1
                elif text[k] == "]":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
        while k < len(text) and text[k] in " \t":
            k += 1
        if k < len(text) and text[k] == "{":
            depth = 0
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
        # Replace the whole thing with blanks so line numbers survive.
        out.append(" " * (k - j) if "\n" not in text[j:k] else re.sub(r"[^\n]", " ", text[j:k]))
        i = k
    return "".join(out)


def strip_latex(text: str, *, preserve_lines: bool = False) -> str:
    """Remove markup, keeping the prose and (optionally) the line structure."""
    text = strip_comments(text)
    for name in DROP_WITH_ARG:
        text = _drop_command_with_arg(text, name)
    for name in KEEP_ARG:
        text = re.sub(r"\\" + name + r"\*?\{", "{", text)
    text = _ENV.sub(" ", text)
    text = text.replace("\\\\", " ")
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)   # remaining control words
    text = re.sub(r"\\[^a-zA-Z]", " ", text)       # escaped punctuation: \% \& \_
    text = text.replace("$", " ").replace("&", " ")
    text = text.replace("~", " ")
    text = re.sub(r"[{}]", "", text)
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    if not preserve_lines:
        text = re.sub(r"[ \t]+", " ", text)
    return text


def normalise_numeric_markup(text: str) -> str:
    """Undo the LaTeX spellings of numbers so tokens compare as plain digits.

    `2{,}127` -> `2127`, `12.4{:}1` -> `12.4:1`, `1\\,000` -> `1000`.
    """
    text = re.sub(r"(\d)\{,\}(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\{:\}(\d)", r"\1:\2", text)
    text = re.sub(r"(\d)\\,(\d)", r"\1\2", text)
    text = re.sub(r"(\d),(\d\d\d)\b", r"\1\2", text)
    return text
