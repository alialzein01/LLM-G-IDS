#!/usr/bin/env python3
"""Rewrite the ladder blocks of the result contracts from the aggregate artifacts.

The schema-5/7 contracts (`results/{unsw_nb15,ton_iot}_current.json`) and the
schema-6 cross-dataset contract were assembled by hand in commit `b03cd0d`, so
until now nothing regenerated them: when the aggregator changed, the contracts
had to be edited field by field and the edit trusted. This script does that
mechanically instead.

It reads `results/multiseed_ladder_v2_head.json` for the canonical
trained-head ladder and `results/multiseed_ladder_v2_legacy.json` for the
superseded prototype-consultant ladder, and writes back only the fields those
artifacts own:

    multi_seed.rungs.<rung>            macro-F1, accuracy and weighted-F1 spreads
    multi_seed.comparisons.<pair>      both bootstrap blocks and both verdicts
    multi_seed.separated_comparisons_two_level
    multi_seed.not_separated_comparisons_two_level
    multi_seed.nothing_separated
    statistical_comparisons.<pair>     a copy of the two-level block
    loop_vs_head_alone_headline        the sentence, with its four numbers
    supersedes.multi_seed.*            the same, from the prototype aggregate

Every other key is preserved, including key order, because the contracts carry
prose, provenance and decisions that no artifact holds. Rung names differ
between the two: the aggregate calls the feedback rung `loop`, the contracts
call it `feedback`.

Run after regenerating the aggregates, then diff the JSON:

    OMP_NUM_THREADS=1 python scripts/refresh_contracts.py
    git diff --stat results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEAD_AGGREGATE = ROOT / "results" / "multiseed_ladder_v2_head.json"
PROTO_AGGREGATE = ROOT / "results" / "multiseed_ladder_v2_legacy.json"
CROSS_CONTRACT = ROOT / "results" / "cross_dataset_comparison.json"
CONTRACTS = {
    "unsw_nb15": ROOT / "results" / "unsw_nb15_current.json",
    "ton_iot": ROOT / "results" / "ton_iot_current.json",
}

# aggregate rung name -> contract rung name
RUNG_NAMES = {"gnn": "gnn", "llm": "llm", "agaf": "agaf",
              "loop": "feedback", "head_alone": "head_alone"}


def _rename(pair: str) -> str:
    """`loop_vs_agaf` -> `feedback_vs_agaf`, leaving every other name alone."""
    a, _, b = pair.partition("_vs_")
    return f"{RUNG_NAMES.get(a, a)}_vs_{RUNG_NAMES.get(b, b)}"


def _contract_name(comparisons: dict, pair: str) -> str | None:
    """Whichever spelling of a comparison this block already uses.

    The current blocks name the feedback rung `feedback`; the superseded blocks
    beneath them were written before the rename and still say `loop`, while
    their rung dictionaries say `feedback`. Matching on both spellings updates
    each block in the name it already carries instead of growing a second copy
    beside it.
    """
    for name in (_rename(pair), pair):
        if name in comparisons:
            return name
    return None


# Keys inside a contract rung or comparison block that the aggregates produce.
# Anything else in those blocks is editorial (`role`, `note`) and is carried
# through untouched.
RUNG_FIELDS = (
    "macro_f1_mean", "macro_f1_std", "macro_f1_per_seed",
    "accuracy_mean", "accuracy_std", "accuracy_per_seed",
    "weighted_f1_mean", "weighted_f1_std", "weighted_f1_per_seed",
)


def _rung_block(existing: dict, rung: dict) -> dict:
    """One rung in contract shape, keeping whatever else the contract wrote there."""
    block = {k: v for k, v in existing.items() if k not in RUNG_FIELDS}
    block["macro_f1_mean"] = rung["mean"]
    block["macro_f1_std"] = rung["std"]
    block["macro_f1_per_seed"] = rung["per_seed"]
    for name in ("accuracy", "weighted_f1"):
        if name not in rung:
            continue
        block[f"{name}_mean"] = rung[name]["mean"]
        block[f"{name}_std"] = rung[name]["std"]
        block[f"{name}_per_seed"] = rung[name]["per_seed"]
    return block


def _separated(block: dict) -> bool:
    return bool(block["ci_low"] > 0.0 or block["ci_high"] < 0.0)


def _comparison_block(existing: dict, comp: dict) -> dict:
    """One comparison in contract shape.

    The two contracts spell the verdict differently: the current blocks carry
    `separated_two_level` and `separated_seed_matched`, the superseded ones a
    single `separated`. Whichever a block already uses is the one that is
    updated, so refreshing does not silently rename a field a test reads.
    """
    block = dict(existing)
    block["per_seed_diff"] = comp["per_seed_diff"]
    block["sign_stable_across_seeds"] = comp["sign_stable_across_seeds"]
    block["two_level"] = comp["two_level"]
    block["seed_matched"] = comp["seed_matched"]
    if "separated" in existing:
        block["separated"] = _separated(comp["two_level"])
    if "separated_two_level" in existing or "separated" not in existing:
        block["separated_two_level"] = _separated(comp["two_level"])
        block["separated_seed_matched"] = _separated(comp["seed_matched"])
    return block


def _headline(comp: dict) -> str:
    """The loop-versus-consultant sentence, rebuilt so its numbers cannot drift."""
    two, matched = comp["two_level"], comp["seed_matched"]
    stable = "stable" if comp["sign_stable_across_seeds"] else "unstable"
    return (
        "The loop is NOT separated from the head alone it consults: two-level "
        f"{two['mean_diff']:+.4f} [{two['ci_low']:+.4f}, {two['ci_high']:+.4f}], "
        f"seed-matched {matched['mean_diff']:+.4f} "
        f"[{matched['ci_low']:+.4f}, {matched['ci_high']:+.4f}], sign {stable} "
        "across seeds. State this wherever the loop's gain over AGAF or the GNN "
        "is stated."
    )


def _apply_ladder(
    multi_seed: dict, agg: dict, changed: list[str], skipped: list[str], where: str
) -> dict:
    """Overwrite one `multi_seed` block in place. Returns the comparison blocks.

    A rung or comparison the aggregate carries but the contract does not declare
    is left out and recorded in `skipped`, never added. Which rungs a contract
    declares is an editorial decision: the prototype-consultant block under
    `supersedes` describes a configuration in which the trained head did not
    exist, so the head-alone rung has no place in it even though the aggregate
    can compute one.
    """

    def record(container: dict, key: str, value, prefix: str = "") -> None:
        if container.get(key) != value:
            changed.append(f"{where}.{prefix}{key}")
        container[key] = value

    for src, dst in RUNG_NAMES.items():
        if src not in agg["rungs"]:
            continue
        if dst not in multi_seed["rungs"]:
            skipped.append(f"{where}.rungs.{dst}")
            continue
        record(multi_seed["rungs"], dst,
               _rung_block(multi_seed["rungs"][dst], agg["rungs"][src]),
               prefix="rungs.")

    comparisons = {}
    for pair, comp in agg["comparisons"].items():
        name = _contract_name(multi_seed["comparisons"], pair)
        if name is None:
            skipped.append(f"{where}.comparisons.{_rename(pair)}")
            continue
        comparisons[name] = _comparison_block(multi_seed["comparisons"][name], comp)
        record(multi_seed["comparisons"], name, comparisons[name], prefix="comparisons.")

    def verdict(block: dict) -> bool:
        return block.get("separated_two_level", block.get("separated", False))

    separated = sorted(n for n, b in comparisons.items() if verdict(b))
    not_separated = sorted(n for n, b in comparisons.items() if not verdict(b))
    if "separated_comparisons_two_level" in multi_seed:
        record(multi_seed, "separated_comparisons_two_level", separated)
    if "not_separated_comparisons_two_level" in multi_seed:
        record(multi_seed, "not_separated_comparisons_two_level", not_separated)
    record(multi_seed, "nothing_separated", not separated)
    return comparisons


def refresh_contract(
    dataset: str, head: dict, proto: dict, changed: list[str], skipped: list[str]
) -> None:
    path = CONTRACTS[dataset]
    payload = json.loads(path.read_text())

    comparisons = _apply_ladder(
        payload["multi_seed"], head[dataset], changed, skipped,
        f"{path.name}:multi_seed",
    )

    for name, block in comparisons.items():
        if payload["statistical_comparisons"].get(name) != block["two_level"]:
            changed.append(f"{path.name}:statistical_comparisons.{name}")
        payload["statistical_comparisons"][name] = block["two_level"]

    if "feedback_vs_head_alone" in comparisons:
        sentence = _headline(comparisons["feedback_vs_head_alone"])
        if payload.get("loop_vs_head_alone_headline") != sentence:
            changed.append(f"{path.name}:loop_vs_head_alone_headline")
        payload["loop_vs_head_alone_headline"] = sentence

    sup = payload.get("supersedes", {})
    if "multi_seed" in sup:
        _apply_ladder(
            sup["multi_seed"], proto[dataset], changed, skipped,
            f"{path.name}:supersedes.multi_seed",
        )

    path.write_text(json.dumps(payload, indent=2) + "\n")


def refresh_cross(head: dict, changed: list[str], skipped: list[str]) -> None:
    payload = json.loads(CROSS_CONTRACT.read_text())
    name = CROSS_CONTRACT.name
    for dataset, block in payload["datasets"].items():
        agg = head[dataset]
        for src, dst in RUNG_NAMES.items():
            if src not in agg["rungs"]:
                continue
            if dst not in block["rungs"]:
                skipped.append(f"{name}:datasets.{dataset}.rungs.{dst}")
                continue
            rung = agg["rungs"][src]
            owned = ("mean", "std", "accuracy_mean", "accuracy_std",
                     "weighted_f1_mean", "weighted_f1_std")
            new = {k: v for k, v in block["rungs"][dst].items() if k not in owned}
            new["mean"] = rung["mean"]
            new["std"] = rung["std"]
            for key in ("accuracy", "weighted_f1"):
                if key in rung:
                    new[f"{key}_mean"] = rung[key]["mean"]
                    new[f"{key}_std"] = rung[key]["std"]
            if block["rungs"][dst] != new:
                changed.append(f"{name}:datasets.{dataset}.rungs.{dst}")
            block["rungs"][dst] = new

        separated = sorted(
            _rename(p) for p, c in agg["comparisons"].items()
            if _separated(c["two_level"])
        )
        if block.get("separated_comparisons_two_level") != separated:
            changed.append(f"{name}:datasets.{dataset}.separated_comparisons_two_level")
        block["separated_comparisons_two_level"] = separated
        if block.get("nothing_separated") != (not separated):
            changed.append(f"{name}:datasets.{dataset}.nothing_separated")
        block["nothing_separated"] = not separated

    CROSS_CONTRACT.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Report what would change and exit non-zero if anything would, "
             "without writing. Use it to prove the contracts match the aggregates.",
    )
    args = parser.parse_args()

    head = json.loads(HEAD_AGGREGATE.read_text())
    proto = json.loads(PROTO_AGGREGATE.read_text())

    if args.check:
        originals = {p: p.read_text() for p in (*CONTRACTS.values(), CROSS_CONTRACT)}

    changed: list[str] = []
    skipped: list[str] = []
    for dataset in CONTRACTS:
        refresh_contract(dataset, head, proto, changed, skipped)
    refresh_cross(head, changed, skipped)

    if args.check:
        for path, text in originals.items():
            path.write_text(text)

    if changed:
        print(f"{len(changed)} field(s) changed:")
        for field in changed:
            print(f"  {field}")
    else:
        print("contracts already match the aggregates; nothing changed")

    if skipped:
        print(f"\n{len(skipped)} field(s) the aggregates carry and the contracts "
              "do not declare, left alone:")
        for field in skipped:
            print(f"  {field}")

    if args.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
