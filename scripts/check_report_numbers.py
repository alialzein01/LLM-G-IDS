#!/usr/bin/env python3
"""Audit the research report's numbers and vocabulary against the artifacts.

Every quantitative claim in `docs/research_report/sections/*.tex` has to trace to
a committed artifact. This script does that mechanically: it pulls every numeric
token out of the sections and checks it against a set built from the result
contracts, the archive, the dataset statistics, and an explicit allowlist for the
protocol constants that live in source code rather than in a JSON file.

It also greps for the vocabulary the report is not allowed to use about its own
system ("significant", "outperform", "top rung", ...) and for the terminology
lock on the semantic branch.

    python scripts/check_report_numbers.py                 # audit all sections
    python scripts/check_report_numbers.py --refresh-stats # recompute dataset stats
    python scripts/check_report_numbers.py --explain 0.8341

Exit code 1 if any number is unknown or any banned term is used; 0 otherwise.
Terminology observations are printed as warnings and do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_audit_common import normalise_numeric_markup, strip_latex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "research_report"
SECTIONS = REPORT / "sections"
RESULTS = ROOT / "results"
ARCHIVE = ROOT / "docs" / "RESULTS_ARCHIVE.md"
STATS_CACHE = REPORT / "dataset_stats.json"
DERIVED = REPORT / "derived_numbers.json"
ALLOWLIST = REPORT / "number_allowlist.txt"

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

# Words the report may not use about its own results. "separated" / "not
# separated" is the locked vocabulary for a difference whose interval excludes
# zero; everything below either overstates that or is promotional.
BANNED_TERMS: dict[str, str] = {
    r"significantl?y?\b": 'use "separated" / "not separated"',
    r"outperform(s|ed|ing)?\b": 'use "higher mean" or "separated above"',
    r"\bbeats?\b": 'use "higher mean" or "separated above"',
    r"top rung": 'use "highest mean"',
    r"\bstate[- ]of[- ]the[- ]art\b": "name the baseline instead",
    r"\brobust(ly|ness)?\b": "say what was measured",
    r"\bsuperior(ity)?\b": "say what was measured",
    r"\bbest\b": 'use "highest mean" unless naming a selection argmax',
}

# Style words banned by the W6 writing standard.
STYLE_TERMS: tuple[str, ...] = (
    r"\bdelve[sd]?\b",
    r"\bleverag(e|es|ed|ing)\b",
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
)

# Fixed phrases where a banned word is a technical term, not a claim about a
# result. Everything else needs an explicit `% numcheck: allow` comment, so the
# exemption is a decision on the record rather than a silent pass.
PHRASE_EXEMPTIONS: tuple[str, ...] = (
    r"best (state|checkpoint|validation|epoch|fold)",
    r"best described",
    r"the best[- ]scoring candidate",
)

# `LLM` is the right word for prior work that uses a generative model. It is the
# wrong word for this system's semantic branch, so it is flagged only where the
# line is also talking about our own components.
OURS = re.compile(
    r"\b(our|we|CySecBERT|semantic branch|semantic head|AGAF|the loop|the head|consultant"
    r"|GNN model|semantic model|fusion model|feedback model)\b",
    re.IGNORECASE,
)

# Terminology lock (warnings, because a legitimate mention can survive review).
TERMINOLOGY_FLAGS: tuple[tuple[str, str], ...] = (
    (r"\bLLM\b", 'prefer "semantic branch" / "trained semantic head" for our own component'),
    (r"the (LLM|model|branch) (reasons|thinks|understands|knows)",
     "the semantic branch is a frozen sentence encoder; it does not reason"),
    (r"generative (large )?language model", "CySecBERT is not generative"),
    (r"\bwe (beat|outperform)\b", "not permitted about our own rungs"),
    (r"\bstatistical(ly)? significan", 'use "separated"'),
    (r"we (are the )?first\b", "absence-of-evidence claims are not verifiable"),
    (r"\bconfirmed unique\b", "absence-of-evidence claims are not verifiable"),
)

# Terms that are fine inside a section heading or a quoted proper name.
HEADING_EXEMPT = re.compile(r"\\(sub)*section\*?\{")


# ---------------------------------------------------------------------------
# the allowed set
# ---------------------------------------------------------------------------


def renderings(value: float) -> set[str]:
    """Every spelling of `value` the report is allowed to use."""
    out: set[str] = set()
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return out
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return out
    for v in {value, abs(value)}:
        if float(v).is_integer() and abs(v) < 1e15:
            out.add(str(int(v)))
        out.add(repr(float(v)).rstrip("0").rstrip(".") if "." in repr(float(v)) else repr(float(v)))
        for places in (2, 3, 4):
            out.add(f"{v:.{places}f}")
        if 0.0 <= abs(v) <= 1.0:
            pct = abs(v) * 100.0
            out.add(f"{pct:.1f}")
            out.add(f"{pct:.2f}")
            out.add(str(int(round(pct))))
    return {t.lstrip("+") for t in out}


# Raw per-edge, per-epoch and per-bootstrap-draw material. A number the report
# cites is a scalar or a short list (per class, per seed, a selection curve), so
# long numeric arrays only widen the allowed set until it stops discriminating.
MAX_ARRAY = 64
BLOCKED_KEYS = frozenset(
    {
        "predictions",
        "probabilities",
        "probs",
        "logits",
        "labels",
        "true_labels",
        "y_true",
        "y_pred",
        "history",
        "training_history",
        "train_loss",
        "val_loss",
        "losses",
        "bootstrap_samples",
        "samples",
        "draws",
        "per_edge",
        "edge_ids",
        "embeddings",
    }
)


def walk_json(node, sink: set[str]) -> None:
    if isinstance(node, dict):
        for key, v in node.items():
            if key in BLOCKED_KEYS:
                continue
            walk_json(v, sink)
    elif isinstance(node, list):
        if len(node) > MAX_ARRAY and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in node
        ):
            return
        for v in node:
            walk_json(v, sink)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        sink |= renderings(node)
    elif isinstance(node, str):
        # Contracts carry prose fields with numbers in them (caveats, verdicts).
        for tok in numeric_tokens(normalise_numeric_markup(node)):
            sink.add(tok)


NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")


def numeric_tokens(text: str) -> list[str]:
    return [m.group(0).lstrip("+-") for m in NUM_RE.finditer(text)]


def significant_digits(token: str) -> int:
    digits = token.replace(".", "").lstrip("0")
    return len(digits.rstrip("0")) if "." in token else len(digits)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"warning: {path.relative_to(ROOT)} is not valid JSON; skipped")
        return None


def load_contract_numbers() -> set[str]:
    """The committed contracts: `results/*.json`, top level only."""
    sink: set[str] = set()
    for path in sorted(RESULTS.glob("*.json")):
        doc = _read_json(path)
        if doc is not None:
            walk_json(doc, sink)
    return sink


def load_run_numbers() -> set[str]:
    """Raw run artifacts under `results/<dir>/`: sweeps, per-run summaries, logs.

    These are kept separate from the contracts on purpose. Together they cover
    about 90% of all four-decimal values in [0, 1], so membership here proves
    almost nothing -- a number that matches only in this pool is reported as
    unverified provenance rather than accepted.
    """
    sink: set[str] = set()
    for path in sorted(RESULTS.rglob("*.json")):
        if path.parent == RESULTS:
            continue
        doc = _read_json(path)
        if doc is not None:
            walk_json(doc, sink)
    return sink


def find_provenance(token: str, limit: int = 40) -> list[str]:
    """Where a token could have come from: `file:dotted.path = value`."""
    hits: list[str] = []

    def walk(node, path: str, rel: str) -> None:
        if len(hits) >= limit:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                if key in BLOCKED_KEYS:
                    continue
                walk(value, f"{path}.{key}" if path else key, rel)
        elif isinstance(node, list):
            if len(node) > MAX_ARRAY and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in node
            ):
                return
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]", rel)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if token in renderings(node):
                hits.append(f"{rel}:{path} = {node}")
        elif isinstance(node, str) and token in numeric_tokens(normalise_numeric_markup(node)):
            hits.append(f"{rel}:{path} (in prose field)")

    for path in sorted(RESULTS.glob("*.json")) + sorted(
        p for p in RESULTS.rglob("*.json") if p.parent != RESULTS
    ):
        doc = _read_json(path)
        if doc is not None:
            walk(doc, "", str(path.relative_to(ROOT)))
        if len(hits) >= limit:
            break
    if ARCHIVE.exists():
        for n, line in enumerate(ARCHIVE.read_text().splitlines(), start=1):
            if token in numeric_tokens(normalise_numeric_markup(line)):
                hits.append(f"docs/RESULTS_ARCHIVE.md:{n}")
                if len(hits) >= limit:
                    break
    return hits


def load_archive_numbers() -> set[str]:
    if not ARCHIVE.exists():
        return set()
    text = normalise_numeric_markup(ARCHIVE.read_text())
    return set(numeric_tokens(text))


def load_derived_numbers() -> set[str]:
    """Arithmetic on contract values, computed by scripts/derive_report_numbers.py."""
    if not DERIVED.exists():
        return set()
    sink: set[str] = set()
    walk_json(json.loads(DERIVED.read_text()), sink)
    return sink


def load_bib_years() -> set[str]:
    bib = REPORT / "references.bib"
    if not bib.exists():
        return set()
    return set(re.findall(r"year\s*=\s*[{\"]?(\d{4})", bib.read_text()))


def load_allowlist() -> tuple[set[str], dict[str, str]]:
    """`value  # why it is allowed` lines. The reason is shown by --explain."""
    values: set[str] = set()
    why: dict[str, str] = {}
    if not ALLOWLIST.exists():
        return values, why
    for raw in ALLOWLIST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token, _, reason = line.partition("#")
        token = token.strip()
        if not token:
            continue
        values.add(token)
        why[token] = reason.strip() or "allowlisted"
    return values, why


# ---------------------------------------------------------------------------
# dataset statistics
# ---------------------------------------------------------------------------


def compute_dataset_stats() -> dict:
    """Edge/node/class counts and imbalance ratios, read off the built graphs."""
    import torch  # imported lazily: the audit runs without it when cached

    sys.path.insert(0, str(ROOT))
    from src.pipeline.common.datasets import DATASETS

    stats: dict[str, dict] = {}
    for key, cfg in DATASETS.items():
        graph_path = ROOT / cfg.graph_path
        if not graph_path.exists():
            continue
        data = torch.load(graph_path, weights_only=False)
        labels = data.edge_label.tolist()
        counts = {i: labels.count(i) for i in range(cfg.num_classes)}
        total = len(labels)
        nonzero = {i: c for i, c in counts.items() if c > 0}
        ratio = max(nonzero.values()) / min(nonzero.values())
        eval_counts = {i: counts[i] for i in cfg.eval_classes}
        stats[key] = {
            "display_name": cfg.display_name,
            "num_nodes": int(data.num_nodes),
            "num_edges": total,
            "num_classes": cfg.num_classes,
            "num_eval_classes": len(cfg.eval_classes),
            "class_counts": {cfg.label_names[i]: c for i, c in counts.items()},
            "class_shares_percent": {
                cfg.label_names[i]: round(100.0 * c / total, 4) for i, c in counts.items()
            },
            "benign_share_percent": round(100.0 * counts[0] / total, 4),
            "imbalance_ratio_all_classes": round(ratio, 4),
            "imbalance_ratio_eval_classes": round(
                max(eval_counts.values()) / min(eval_counts.values()), 4
            ),
            "smallest_class_edges": min(nonzero.values()),
            "largest_class_edges": max(nonzero.values()),
            "attack_classes_at_or_below_35": sum(
                1 for i, c in counts.items() if i != 0 and 0 < c <= 35
            ),
            "attack_classes_at_or_below_40": sum(
                1 for i, c in counts.items() if i != 0 and 0 < c <= 40
            ),
            "dropped_class_edges": {cfg.label_names[i]: counts[i] for i in cfg.dropped_classes},
            "num_edge_features": int(data.edge_attr.shape[1]),
            "num_node_features": int(data.x.shape[1]),
        }
        for attr in ("num_protocols", "num_ports"):
            if hasattr(data, attr):
                stats[key][attr] = int(getattr(data, attr))
    if "unsw_nb15" in stats and "ton_iot" in stats:
        stats["cross_dataset"] = {
            "imbalance_ratio_of_ratios": round(
                stats["ton_iot"]["imbalance_ratio_all_classes"]
                / stats["unsw_nb15"]["imbalance_ratio_all_classes"],
                4,
            )
        }
    return stats


def load_dataset_stats(refresh: bool) -> tuple[dict, set[str]]:
    if refresh or not STATS_CACHE.exists():
        stats = compute_dataset_stats()
        if stats:
            STATS_CACHE.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    if not STATS_CACHE.exists():
        return {}, set()
    stats = json.loads(STATS_CACHE.read_text())
    sink: set[str] = set()
    walk_json(stats, sink)
    return stats, sink


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


def exempt_lines(raw: str) -> set[int]:
    """Lines carrying a `% numcheck: allow` comment are skipped."""
    out = set()
    for n, line in enumerate(raw.splitlines(), start=1):
        if re.search(r"%\s*numcheck:\s*allow", line):
            out.add(n)
    return out


def audit_numbers(
    path: Path, allowed: set[str], run_only: set[str]
) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    """Returns (unknown, run-artifact-only) numbers, each as (line, token, context)."""
    raw = path.read_text()
    skip = exempt_lines(raw)
    stripped = strip_latex(raw, preserve_lines=True)
    unknown: list[tuple[int, str, str]] = []
    unverified: list[tuple[int, str, str]] = []
    for n, line in enumerate(stripped.splitlines(), start=1):
        if n in skip:
            continue
        norm = normalise_numeric_markup(line)
        # Cross-references and identifiers are not quantitative claims.
        norm = re.sub(r"\b(RQ|D|Section|Table|Figure|Appendix|§)\s*\d+", " ", norm)
        for m in NUM_RE.finditer(norm):
            token = m.group(0).lstrip("+-")
            if significant_digits(token) < 2:
                continue
            if token in allowed:
                continue
            context = norm.strip()[:110]
            if token in run_only:
                unverified.append((n, token, context))
            else:
                unknown.append((n, token, context))
    return unknown, unverified


def audit_words(path: Path) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
    raw = path.read_text()
    skip = exempt_lines(raw)
    lines = strip_latex(raw, preserve_lines=True).splitlines()
    raw_lines = raw.splitlines()
    errors: list[tuple[int, str, str]] = []
    warnings: list[tuple[int, str, str]] = []
    for n, line in enumerate(lines, start=1):
        if n in skip:
            continue
        heading = bool(HEADING_EXEMPT.search(raw_lines[n - 1])) if n <= len(raw_lines) else False
        exempt_spans = [
            m.span()
            for pattern in PHRASE_EXEMPTIONS
            for m in re.finditer(pattern, line, flags=re.IGNORECASE)
        ]

        def inside_exemption(span: tuple[int, int]) -> bool:
            return any(a <= span[0] and span[1] <= b for a, b in exempt_spans)

        for pattern, advice in BANNED_TERMS.items():
            for m in re.finditer(pattern, line, flags=re.IGNORECASE):
                if heading or inside_exemption(m.span()):
                    continue
                errors.append((n, m.group(0), advice))
        for pattern in STYLE_TERMS:
            for m in re.finditer(pattern, line, flags=re.IGNORECASE):
                errors.append((n, m.group(0), "banned by the writing standard"))
        for pattern, advice in TERMINOLOGY_FLAGS:
            for m in re.finditer(pattern, line, flags=re.IGNORECASE):
                if m.group(0) == "LLM" and not OURS.search(line):
                    continue
                warnings.append((n, m.group(0), advice))
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections", nargs="*", help="section .tex files (default: all)")
    ap.add_argument("--refresh-stats", action="store_true", help="recompute dataset_stats.json")
    ap.add_argument("--explain", metavar="TOKEN", help="show every artifact a number could come from")
    ap.add_argument("--quiet", action="store_true", help="print only the summary and failures")
    ap.add_argument(
        "--strict-provenance",
        action="store_true",
        help="also fail on numbers matched only in raw run artifacts",
    )
    args = ap.parse_args()

    contract = load_contract_numbers()
    archive = load_archive_numbers()
    _, stats_numbers = load_dataset_stats(args.refresh_stats)
    allowlisted, allow_why = load_allowlist()
    derived = load_derived_numbers()
    years = load_bib_years()
    allowed = contract | archive | stats_numbers | derived | allowlisted | years
    run_only = load_run_numbers() - allowed

    if args.explain:
        token = args.explain.lstrip("+-")
        sources = [
            name
            for name, pool in (
                ("results/*.json contracts", contract),
                ("RESULTS_ARCHIVE.md", archive),
                ("dataset_stats.json", stats_numbers),
                ("derived_numbers.json", derived),
                ("number_allowlist.txt", allowlisted),
                ("references.bib years", years),
                ("raw run artifacts only", run_only),
            )
            if token in pool
        ]
        print(f"{token}: {', '.join(sources) if sources else 'NOT in any pool'}")
        if token in allow_why:
            print(f"  allowlist reason: {allow_why[token]}")
        for hit in find_provenance(token):
            print(f"  {hit}")
        return 0 if sources else 1

    paths = [Path(p) for p in args.sections] if args.sections else sorted(SECTIONS.glob("*.tex"))
    paths = [p if p.is_absolute() else ROOT / p for p in paths]

    if not args.quiet:
        four_dp = sum(1 for t in allowed if re.fullmatch(r"0\.\d{4}", t))
        print(
            f"allowed set: {len(allowed)} tokens "
            f"({len(contract)} contracts, {len(archive)} archive, "
            f"{len(stats_numbers)} dataset stats, {len(derived)} derived, "
            f"{len(allowlisted)} allowlist, {len(years)} bib years)"
        )
        print(
            f"discrimination: {four_dp}/10000 of the 0.xxxx range is accepted "
            f"({four_dp / 100:.1f}% of four-decimal values in [0,1]); "
            f"{len(run_only)} further tokens exist only in raw run artifacts"
        )

    unknown_total = word_total = warn_total = unverified_total = 0
    for path in paths:
        if not path.exists():
            print(f"missing: {path}")
            return 2
        rel = path.relative_to(ROOT)
        unknown, unverified = audit_numbers(path, allowed, run_only)
        errors, warnings = audit_words(path)
        unknown_total += len(unknown)
        unverified_total += len(unverified)
        word_total += len(errors)
        warn_total += len(warnings)
        failed = bool(unknown) or bool(errors) or (args.strict_provenance and unverified)
        status = "FAIL" if failed else "OK"
        if args.quiet and status == "OK" and not warnings and not unverified:
            continue
        print(
            f"\n{rel}  [{status}]  unknown={len(unknown)} unverified={len(unverified)} "
            f"banned={len(errors)} warn={len(warnings)}"
        )
        for n, token, context in unknown:
            print(f"  L{n}: unknown number {token!r}  |  {context}")
        for n, token, context in unverified:
            print(f"  L{n}: {token!r} appears only in a raw run artifact, not in a contract "
                  f"or the archive -- run --explain {token}  |  {context}")
        for n, term, advice in errors:
            print(f"  L{n}: banned term {term!r} -- {advice}")
        for n, term, advice in warnings:
            print(f"  L{n}: check {term!r} -- {advice}")

    print(
        f"\nsummary: {unknown_total} unknown number(s), {unverified_total} unverified-provenance, "
        f"{word_total} banned term(s), {warn_total} terminology warning(s) "
        f"across {len(paths)} section(s)"
    )
    failing = unknown_total or word_total or (args.strict_provenance and unverified_total)
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
