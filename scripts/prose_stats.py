#!/usr/bin/env python3
"""Measure the report's prose, section by section.

The W6 writing pass has numeric targets: sentence-length mean 15-22 words with a
standard deviation of at least 7, at least a few short sentences, no em dashes,
few semicolons, varied paragraph lengths. This script measures all of that so a
before/after comparison can go in the commit message instead of an impression.

    python scripts/prose_stats.py                       # every section
    python scripts/prose_stats.py --json                # machine-readable
    python scripts/prose_stats.py sections/04_results.tex
    python scripts/prose_stats.py --baseline before.json  # show the deltas
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_audit_common import strip_latex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "docs" / "research_report" / "sections"

TARGETS = {
    "sentence_len_mean": (15.0, 22.0),
    "sentence_len_std": (7.0, None),
    "short_sentence_share": (0.05, None),   # sentences under 8 words
    "long_sentence_share": (None, 0.10),    # sentences over 35 words
    "em_dashes": (None, 0),
    "semicolons_per_1000_words": (None, 2.0),
}

FLAGGED_TERMS: tuple[str, ...] = (
    r"\bdelve[sd]?\b",
    r"\bleverag(e|es|ed|ing)\b",
    r"\brobust(ly|ness)?\b",
    r"\bcrucial(ly)?\b",
    r"\bpivotal\b",
    r"\blandscape\b",
    r"\bcomprehensive(ly)?\b",
    r"\bnuanced\b",
    r"\bshowcas(e|es|ed|ing)\b",
    r"\bunderscor(e|es|ed|ing)\b",
    r"\btestament\b",
    r"\bholistic(ally)?\b",
    r"it is worth noting",
    r"it should be mentioned",
    r"\bin order to\b",
    r"\bnotably\b",
    r"\binterestingly\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\butilis[ez]",
    r"\bsignificantl?y?\b",
    r"\boutperform(s|ed|ing)?\b",
)

# Abbreviations that end in a period without ending a sentence.
ABBREV = (
    "e.g.", "i.e.", "et al.", "cf.", "vs.", "Fig.", "Eq.", "Sec.", "Tab.",
    "Dr.", "Prof.", "approx.", "resp.", "no.", "Nos.",
)

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z(\"'\\$])")


def to_prose(text: str) -> str:
    """LaTeX in, plain paragraphs out. Tables and equations are dropped."""
    # Whole environments whose content is not prose.
    for env in ("tabular", "tabularx", "equation", "align", "verbatim", "lstlisting", "tikzpicture"):
        text = re.sub(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", " ", text, flags=re.DOTALL
        )
    # Headings are labels, not sentences, and `\item[RQ1.]` labels are not
    # either. Counting them drags the short-sentence share up for free.
    text = re.sub(r"\\(sub)*section\*?\{[^}]*\}", "\n\n", text)
    text = re.sub(r"\\item\s*\[[^\]]*\]", r"\\item ", text)
    # Run-in headings (`\textbf{Selection.}` opening a paragraph) are labels too.
    text = re.sub(r"\\(paragraph|subparagraph|textbf|emph|textit)\*?\{([A-Z][^{}]{0,40}\.)\}\s*", " ", text)
    text = strip_latex(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def paragraphs(prose: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", prose) if p.strip()]


def sentences(paragraph: str) -> list[str]:
    flat = re.sub(r"\s+", " ", paragraph).strip()
    for abbr in ABBREV:
        flat = flat.replace(abbr, abbr.replace(".", "\u0001"))
    parts = [s.strip() for s in _SENT_END.split(flat) if s.strip()]
    return [p.replace("\u0001", ".") for p in parts]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-.:/]*", text)


def histogram(lengths: list[int]) -> dict[str, int]:
    bins = {"<8": 0, "8-14": 0, "15-22": 0, "23-30": 0, "31-35": 0, ">35": 0}
    for n in lengths:
        if n < 8:
            bins["<8"] += 1
        elif n < 15:
            bins["8-14"] += 1
        elif n < 23:
            bins["15-22"] += 1
        elif n < 31:
            bins["23-30"] += 1
        elif n <= 35:
            bins["31-35"] += 1
        else:
            bins[">35"] += 1
    return bins


def monotony_runs(lengths: list[int]) -> int:
    """How many places three consecutive sentences sit within +/-3 words."""
    runs = 0
    for i in range(len(lengths) - 2):
        window = lengths[i : i + 3]
        if max(window) - min(window) <= 3:
            runs += 1
    return runs


def analyse(path: Path) -> dict:
    raw = path.read_text()
    prose = to_prose(raw)
    paras = paragraphs(prose)
    per_para = [sentences(p) for p in paras]
    sents = [s for group in per_para for s in group]
    lengths = [len(words(s)) for s in sents]
    total_words = sum(lengths)
    flagged: dict[str, int] = {}
    for pattern in FLAGGED_TERMS:
        hits = len(re.findall(pattern, prose, flags=re.IGNORECASE))
        if hits:
            flagged[pattern] = hits
    para_lengths = [len(group) for group in per_para if group]
    return {
        "file": str(path.relative_to(ROOT)),
        "words": total_words,
        "sentences": len(sents),
        "paragraphs": len(para_lengths),
        "sentence_len_mean": round(statistics.fmean(lengths), 2) if lengths else 0.0,
        "sentence_len_std": round(statistics.pstdev(lengths), 2) if len(lengths) > 1 else 0.0,
        "sentence_len_median": round(statistics.median(lengths), 1) if lengths else 0.0,
        "sentence_len_min": min(lengths) if lengths else 0,
        "sentence_len_max": max(lengths) if lengths else 0,
        "short_sentence_share": round(sum(1 for n in lengths if n < 8) / len(lengths), 4) if lengths else 0.0,
        "long_sentence_share": round(sum(1 for n in lengths if n > 35) / len(lengths), 4) if lengths else 0.0,
        "histogram": histogram(lengths),
        "monotony_runs": monotony_runs(lengths),
        "paragraph_len_mean": round(statistics.fmean(para_lengths), 2) if para_lengths else 0.0,
        "paragraph_len_min": min(para_lengths) if para_lengths else 0,
        "paragraph_len_max": max(para_lengths) if para_lengths else 0,
        "paragraph_lengths": para_lengths,
        "em_dashes": raw.count("---") + raw.count("\u2014"),
        "semicolons": prose.count(";"),
        "semicolons_per_1000_words": round(1000.0 * prose.count(";") / total_words, 2) if total_words else 0.0,
        "colon_list_openers": len(re.findall(r":\s*\n?\s*\\begin\{(itemize|enumerate|description)\}", raw)),
        "flagged_terms": flagged,
    }


def off_target(stats: dict) -> list[str]:
    misses = []
    for key, (lo, hi) in TARGETS.items():
        value = stats.get(key)
        if value is None:
            continue
        if lo is not None and value < lo:
            misses.append(f"{key}={value} < {lo}")
        if hi is not None and value > hi:
            misses.append(f"{key}={value} > {hi}")
    return misses


def render(stats: dict, baseline: dict | None) -> None:
    def delta(key: str) -> str:
        if not baseline or key not in baseline:
            return ""
        before, now = baseline[key], stats[key]
        if not isinstance(before, (int, float)) or not isinstance(now, (int, float)):
            return ""
        diff = now - before
        return f"  ({before} -> {now}, {diff:+.2f})" if diff else "  (unchanged)"

    print(f"\n{stats['file']}")
    print(f"  {stats['words']} words, {stats['sentences']} sentences, {stats['paragraphs']} paragraphs")
    print(
        f"  sentence length: mean {stats['sentence_len_mean']}{delta('sentence_len_mean')}, "
        f"std {stats['sentence_len_std']}{delta('sentence_len_std')}, "
        f"median {stats['sentence_len_median']}, range {stats['sentence_len_min']}-{stats['sentence_len_max']}"
    )
    hist = stats["histogram"]
    width = max(hist.values()) or 1
    for label, count in hist.items():
        bar = "#" * int(round(24 * count / width))
        print(f"    {label:>6} | {bar:<24} {count}")
    print(
        f"  short (<8 words): {stats['short_sentence_share']:.1%}   "
        f"long (>35): {stats['long_sentence_share']:.1%}   "
        f"3-in-a-row within +/-3 words: {stats['monotony_runs']}"
    )
    print(
        f"  paragraphs: mean {stats['paragraph_len_mean']} sentences, "
        f"range {stats['paragraph_len_min']}-{stats['paragraph_len_max']}"
    )
    print(
        f"  em dashes: {stats['em_dashes']}{delta('em_dashes')}   "
        f"semicolons: {stats['semicolons']} ({stats['semicolons_per_1000_words']}/1000 words)   "
        f"colon-list openers: {stats['colon_list_openers']}"
    )
    if stats["flagged_terms"]:
        terms = ", ".join(f"{p} x{n}" for p, n in sorted(stats["flagged_terms"].items()))
        print(f"  flagged terms: {terms}")
    misses = off_target(stats)
    print(f"  off target: {'; '.join(misses) if misses else 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections", nargs="*", help="section .tex files (default: all)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    ap.add_argument("--baseline", metavar="FILE", help="a previous --json run, to show deltas")
    args = ap.parse_args()

    paths = [Path(p) for p in args.sections] if args.sections else sorted(SECTIONS.glob("*.tex"))
    paths = [p if p.is_absolute() else ROOT / p for p in paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing: " + ", ".join(str(p) for p in missing))
        return 2

    results = [analyse(p) for p in paths]
    if args.json:
        print(json.dumps({r["file"]: r for r in results}, indent=2))
        return 0

    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    for stats in results:
        render(stats, (baseline or {}).get(stats["file"]))

    total_words = sum(r["words"] for r in results)
    total_sents = sum(r["sentences"] for r in results)
    print(f"\ntotal: {total_words} words, {total_sents} sentences across {len(results)} section(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
