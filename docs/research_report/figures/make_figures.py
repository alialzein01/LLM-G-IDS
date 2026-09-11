#!/usr/bin/env python3
"""Generate the report's data figures from the committed contracts.

Six figures. Every number is read from `results/*.json`, `dataset_stats.json` or
`derived_numbers.json`, so a figure cannot drift from the text: change a
contract, re-run this, and the plot follows. The three conceptual diagrams
(`fig_architecture`, `fig_loop_flow`, `fig_ceiling`) are drawn externally and
are not produced here; `FIGURE_BRIEFS.md` carries their specifications.

    python docs/research_report/figures/make_figures.py
    python docs/research_report/figures/make_figures.py --only ladder mechanism

Two rules are enforced rather than remembered. The two datasets never share a
y-axis, because they score different numbers of classes and their levels are not
comparable. And every error bar is training-seed variance at a fixed fold
partition, which each caption in the report repeats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT = ROOT / "docs" / "research_report"

# Okabe-Ito, colourblind-safe. One role per colour, fixed across every figure.
GRAPH = "#0072B2"
SEMANTIC = "#E69F00"
AGAF = "#009E73"
LOOP = "#D55E00"
ORACLE = "#CC79A7"
NEUTRAL = "#999999"

WIDTH = 6.1  # inches: the report's text block is 15.6 cm

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def save(fig, name: str) -> None:
    out = HERE / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------


def fig_ladder() -> None:
    """Four rungs per dataset, with the loop shown under both consultants."""
    head = load("results/multiseed_ladder_v2_head.json")
    proto = load("results/multiseed_ladder_v2_legacy.json")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.2))
    panels = [
        ("unsw_nb15", "NF-UNSW-NB15 (10 classes)", (0.20, 0.94)),
        ("ton_iot", "NF-ToN-IoT (8 classes)", (0.20, 0.60)),
    ]
    for ax, (key, title, ylim) in zip(axes, panels):
        h, p = head[key]["rungs"], proto[key]["rungs"]
        bars = [
            ("graph encoder", h["gnn"], GRAPH, False),
            ("semantic rung", h["llm"], SEMANTIC, False),
            ("AGAF", h["agaf"], AGAF, False),
            ("loop, prototype", p["loop"], LOOP, True),
            ("loop, trained head", h["loop"], LOOP, False),
        ]
        for i, (label, rung, colour, hatched) in enumerate(bars):
            ax.bar(
                i,
                rung["mean"],
                width=0.66,
                color="white" if hatched else colour,
                edgecolor=colour,
                hatch="////" if hatched else None,
                linewidth=1.0,
                yerr=rung["std"] if rung["std"] > 0 else None,
                capsize=2.5,
                error_kw={"linewidth": 0.9, "ecolor": "#333333"},
                zorder=2,
            )
            ax.scatter(
                [i] * 3,
                list(rung["per_seed"].values()),
                s=9,
                facecolors="none",
                edgecolors="#333333",
                linewidths=0.7,
                zorder=3,
            )
        # The semantic rung is deterministic; say so rather than drawing nothing.
        ax.annotate(
            "deterministic",
            xy=(1, h["llm"]["mean"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color="#555555",
        )
        # The consultant change, drawn between the two loop bars.
        lo, hi = p["loop"]["mean"], h["loop"]["mean"]
        ax.annotate(
            "",
            xy=(4, hi),
            xytext=(3, lo),
            arrowprops={"arrowstyle": "->", "color": LOOP, "linewidth": 1.0},
        )
        ax.text(
            3.5,
            (lo + hi) / 2,
            f"{hi - lo:+.4f}",
            fontsize=7,
            color=LOOP,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none"},
        )
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=7, rotation=28, ha="right",
                           rotation_mode="anchor")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("pooled out-of-fold macro-F1")
    fig.text(
        0.5,
        -0.10,
        "Error bars are training-seed variance at a fixed fold partition; open circles are the "
        "three seeds.\nThe two panels have independent axes: the datasets score different "
        "numbers of classes and their levels are not comparable.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "fig_ladder")


def fig_imbalance() -> None:
    """Class sizes on a log axis; the report's organising contrast."""
    stats = load("docs/research_report/dataset_stats.json")
    dropped = {"dos", "ransomware"}

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1))
    for ax, key in zip(axes, ("unsw_nb15", "ton_iot")):
        counts = stats[key]["class_counts"]
        order = sorted(counts.items(), key=lambda kv: -kv[1])
        names = [n for n, _ in order]
        values = [v for _, v in order]
        ypos = range(len(names))
        for y, (name, value) in zip(ypos, order):
            excluded = key == "ton_iot" and name in dropped
            ax.barh(
                y,
                value,
                height=0.68,
                color="white" if excluded else GRAPH,
                edgecolor=GRAPH,
                linestyle="--" if excluded else "-",
                linewidth=0.9,
                zorder=3,
            )
            if value <= 35:
                ax.axhspan(y - 0.42, y + 0.42, color="#D55E00", alpha=0.09, zorder=1)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1, 3000)
        ax.set_xlabel("edges (log scale)")
        display = {"unsw_nb15": "NF-UNSW-NB15", "ton_iot": "NF-ToN-IoT"}[key]
        ax.set_title(f"{display}, {stats[key]['num_edges']:,} edges", fontsize=9)
        ax.grid(axis="x", linewidth=0.4, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.text(
            0.97,
            0.52,
            f"{stats[key]['imbalance_ratio_all_classes']:g} : 1",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#888888"},
        )
    axes[1].annotate(
        "excluded from the metric,\nkept for message passing",
        xy=(4.2, 8.0),
        xytext=(40, 6.6),
        fontsize=6.5,
        color="#444444",
        arrowprops={"arrowstyle": "->", "color": "#888888", "linewidth": 0.7},
    )
    handles = [Patch(facecolor="#D55E00", alpha=0.15, label="35 edges or fewer")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               frameon=False, fontsize=7)
    ratio = stats["cross_dataset"]["imbalance_ratio_of_ratios"]
    fig.text(
        0.5,
        -0.08,
        f"The two graphs differ by roughly a factor of {ratio:.0f} in imbalance severity.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "fig_imbalance")


def fig_mechanism() -> None:
    """The isolation experiment: which arms clear zero with fusion disabled."""
    rows = [
        ("oracle, edge", "oracle_edge_vs_control_head_only", ORACLE, True),
        ("trained head, edge", "head_trained_edge_vs_control_head_only", LOOP, True),
        ("prototype, edge", "real_prototype_edge_vs_control_head_only", NEUTRAL, True),
        ("oracle, attention", "oracle_attention_vs_control_head_only", ORACLE, False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    panels = [
        ("unsw_nb15", "NF-UNSW-NB15", (-0.07, 0.10)),
        ("ton_iot", "NF-ToN-IoT", (-0.10, 0.21)),
    ]
    for ax, (key, title, xlim) in zip(axes, panels):
        paired = load(f"results/{key}_oracle_ceiling_v2_trained_head.json")["paired_comparisons"]
        ax.axvspan(-0.012, 0.012, color="#BBBBBB", alpha=0.30, zorder=0)
        ax.axvline(0, color="#222222", linewidth=1.1, zorder=1)
        for i, (label, field, colour, filled) in enumerate(rows):
            c = paired[field]
            y = len(rows) - 1 - i
            separated = c["ci_low"] > 0 or c["ci_high"] < 0
            ax.plot(
                [c["ci_low"], c["ci_high"]],
                [y, y],
                color=colour,
                linewidth=1.3,
                solid_capstyle="butt",
                zorder=2,
            )
            for x in (c["ci_low"], c["ci_high"]):
                ax.plot([x, x], [y - 0.13, y + 0.13], color=colour, linewidth=1.1, zorder=2)
            ax.scatter(
                [c["mean_diff"]],
                [y],
                s=26,
                marker="o" if filled else "s",
                facecolors=colour if separated else "white",
                edgecolors=colour,
                linewidths=1.1,
                zorder=3,
            )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=7.5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.set_xlabel("paired difference against control")
        ax.grid(axis="x", linewidth=0.4, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
    handles = [
        Patch(facecolor="#BBBBBB", alpha=0.30, label="run-to-run drift without thread pinning"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=6.5)
    fig.text(
        0.5,
        -0.10,
        "Output fusion is disabled in every arm, so advice reaches the classifier only through "
        "the injection path.\nFilled markers mark intervals that exclude zero. Only the oracle "
        "converts the channel; on NF-ToN-IoT\nthe trained-head consultant is separated in the "
        "wrong direction.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "fig_mechanism")


def fig_decomposition() -> None:
    """Where the NF-UNSW-NB15 margin against a published architecture comes from."""
    d = load("docs/research_report/derived_numbers.json")["decomposition_unsw"]
    levels, endpoint = d["levels"], d["endpoint_loop"]
    base = levels["te_g_sage_as_published"]
    steps = [
        ("fair tuning", d["step_tuning"], endpoint["share_tuning_percent"], "#7FB8DC"),
        ("our ten node centralities", d["step_node_features"], endpoint["share_node_features_percent"], GRAPH),
        ("architecture", endpoint["step_architecture"], endpoint["share_architecture_percent"], LOOP),
    ]

    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    ax.bar(0, base, width=0.62, color=NEUTRAL, zorder=2)
    ax.text(0, base + 0.008, f"{base:.4f}", ha="center", fontsize=7.5)

    running = base
    for i, (label, delta, share, colour) in enumerate(steps, start=1):
        ax.bar(i, delta, bottom=running, width=0.62, color=colour, zorder=2)
        ax.plot([i - 0.31 - 0.38, i - 0.31], [running, running], color="#666666",
                linewidth=0.7, linestyle=":", zorder=1)
        ax.text(i, running + delta / 2, f"{delta:+.4f}\n{share}%", ha="center", va="center",
                fontsize=7, color="white" if colour != "#7FB8DC" else "#222222")
        running += delta
    ax.bar(len(steps) + 1, running, width=0.62, color=LOOP, zorder=2)
    ax.text(len(steps) + 1, running + 0.008, f"{running:.4f}", ha="center", fontsize=7.5)

    agaf = levels["agaf"]
    ax.plot([len(steps) - 0.31, len(steps) + 1.31], [agaf, agaf], color=AGAF,
            linewidth=1.1, linestyle="--", zorder=3)
    ax.text(len(steps) + 1.34, agaf, f"AGAF rung {agaf:.4f}", fontsize=7, color=AGAF,
            va="center", ha="left")

    ax.set_xticks(range(len(steps) + 2))
    ax.set_xticklabels(
        ["TE-G-SAGE as published"] + [s[0] for s in steps] + ["our loop rung"],
        fontsize=7, rotation=20, ha="right", rotation_mode="anchor",
    )
    ax.set_ylim(0.30, 0.92)
    ax.set_xlim(-0.6, len(steps) + 2.5)
    ax.set_ylabel("pooled out-of-fold macro-F1")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        -0.06,
        "Point estimates only: every level is a three-seed mean and no step carries an interval "
        "or a separation claim.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "fig_decomposition")


def fig_perclass() -> None:
    """Per-class differences. Levels are already tabulated; the deltas are the argument."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.pipeline.common.datasets import DATASETS

    per_class = load("results/multiseed_head_per_class.json")
    stats = load("docs/research_report/dataset_stats.json")

    labels, sizes, values, block_edges = [], [], [], []
    for key in ("unsw_nb15", "ton_iot"):
        cfg = DATASETS[key]
        block = per_class[key]["per_class_f1_3seed"]
        counts = stats[key]["class_counts"]
        block_edges.append(len(labels))
        for j, ci in enumerate(per_class[key]["eval_classes"]):
            name = cfg.label_names[ci]
            g = block["gnn"]["mean"][j]
            a = block["agaf"]["mean"][j]
            loop = block["loop"]["mean"][j]
            labels.append(name)
            sizes.append(counts[name])
            values.append([loop - a, loop - g])

    limit = 0.22
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["loop $-$ AGAF", "loop $-$ graph encoder"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{n}  ({s})" for n, s in zip(labels, sizes)], fontsize=7.5)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center", fontsize=6.8,
                    color="#111111" if abs(v) < 0.13 else "white")
            if v < 0:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor="#111111", linewidth=1.2)
                )
    for edge in block_edges[1:]:
        ax.axhline(edge - 0.5, color="#222222", linewidth=1.6)
    spans = block_edges + [len(labels)]
    for name, lo, hi in zip(("NF-UNSW-NB15", "NF-ToN-IoT"), spans[:-1], spans[1:]):
        ax.text(-0.42, (lo + hi - 1) / 2, name, rotation=90, va="center", ha="center",
                fontsize=8, transform=ax.get_yaxis_transform())
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("difference in per-class F1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Per-class differences, three-seed means (class size in brackets)", fontsize=9)
    fig.text(
        0.5,
        -0.02,
        "Cells outlined in black are losses. No per-class difference carries an interval, and on "
        "classes of 12 to 40 edges\na single edge moves F1 substantially.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "fig_perclass")


def fig_comparisons() -> None:
    """Which ladder steps are separated, under each consultant, under each bootstrap."""
    head = load("results/multiseed_ladder_v2_head.json")
    proto = load("results/multiseed_ladder_v2_legacy.json")

    spec = {
        "unsw_nb15": [
            ("loop $-$ semantic", head, "loop_vs_llm", LOOP, "trained head"),
            ("loop $-$ graph", head, "loop_vs_gnn", LOOP, "trained head"),
            ("loop $-$ AGAF", head, "loop_vs_agaf", LOOP, "trained head"),
            ("loop $-$ graph", proto, "loop_vs_gnn", "#F0A882", "prototype"),
            ("loop $-$ AGAF", proto, "loop_vs_agaf", "#F0A882", "prototype"),
            ("AGAF $-$ graph", head, "agaf_vs_gnn", AGAF, "either"),
        ],
        "ton_iot": [
            ("loop $-$ semantic", head, "loop_vs_llm", LOOP, "trained head"),
            ("loop $-$ AGAF", head, "loop_vs_agaf", LOOP, "trained head"),
            ("loop $-$ graph", head, "loop_vs_gnn", LOOP, "trained head"),
            ("loop $-$ AGAF", proto, "loop_vs_agaf", "#F0A882", "prototype"),
            ("loop $-$ graph", proto, "loop_vs_gnn", "#F0A882", "prototype"),
            ("AGAF $-$ graph", head, "agaf_vs_gnn", AGAF, "either"),
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
    for ax, (key, title, xlim) in zip(
        axes,
        [("unsw_nb15", "NF-UNSW-NB15", (-0.08, 0.16)), ("ton_iot", "NF-ToN-IoT", (-0.12, 0.31))],
    ):
        rows = spec[key]
        ax.axvline(0, color="#222222", linewidth=1.1, zorder=1)
        ticks, ticklabels = [], []
        for i, (label, source, field, colour, group) in enumerate(rows):
            y = len(rows) - 1 - i
            comp = source[key]["comparisons"][field]
            for offset, kind, marker in ((0.14, "two_level", "o"), (-0.14, "seed_matched", "s")):
                c = comp[kind]
                separated = c["ci_low"] > 0 or c["ci_high"] < 0
                alpha = 1.0 if kind == "two_level" else 0.55
                ax.plot([c["ci_low"], c["ci_high"]], [y + offset, y + offset],
                        color=colour, linewidth=1.2, alpha=alpha, zorder=2)
                ax.scatter([c["mean_diff"]], [y + offset], s=20, marker=marker,
                           facecolors=colour if separated else "white",
                           edgecolors=colour, linewidths=1.0, alpha=alpha, zorder=3)
            ticks.append(y)
            ticklabels.append(f"{label}\n({group})")
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, fontsize=6.8)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.set_xlabel("difference in pooled out-of-fold macro-F1")
        ax.grid(axis="x", linewidth=0.4, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], color="#444444", marker="o", linestyle="-",
                   label="two-level (edges and training seed)"),
        plt.Line2D([], [], color="#444444", marker="s", linestyle="-", alpha=0.55,
                   label="seed-matched (edges only)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=2, frameon=False, fontsize=6.5)
    fig.text(
        0.5,
        -0.13,
        "Filled markers mark intervals that exclude zero. Every loop comparison clears zero "
        "under the trained-head\nconsultant and none does under the prototype. The two-level "
        "interval is the one the decision rule uses.",
        ha="center",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "fig_comparisons")


FIGURES = {
    "ladder": fig_ladder,
    "imbalance": fig_imbalance,
    "mechanism": fig_mechanism,
    "decomposition": fig_decomposition,
    "perclass": fig_perclass,
    "comparisons": fig_comparisons,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="a subset to rebuild")
    args = ap.parse_args()
    names = args.only or list(FIGURES)
    print(f"generating {len(names)} figure(s)")
    for name in names:
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
