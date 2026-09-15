#!/usr/bin/env python3
"""Data figures for the M2 research report.

Every value below is transcribed from the frozen figure briefs in
`figures/FIGURE_BRIEFS.md`, which were themselves checked against the result
contracts under `results/`. Nothing here is computed at draw time, so a plot
cannot drift from the tables; it can only disagree, and `check_report_numbers.py`
is what catches that.

Usage:  OMP_NUM_THREADS=1 python make_plots.py [name ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent.parent

# Okabe-Ito; roles are fixed across all nine figures.
C_GRAPH, C_SEM, C_FUSE = "#0072B2", "#E69F00", "#009E73"
C_LOOP, C_ORACLE, C_NEUT = "#D55E00", "#CC79A7", "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.color": "#cccccc",
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MINUS = "$-$"  # cmr10 has no U+2212; mathtext supplies it


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")


# --------------------------------------------------------------------------- 3
def fig_ladder():
    panels = [
        ("NF-UNSW-NB15 (10 classes)", (0.20, 0.90),
         [0.7437, 0.7353, 0.7793, 0.7644, 0.8341],
         [0.0228, None, 0.0116, None, 0.0042],
         {0: (0.7219, 0.7674, 0.7418), 2: (0.7680, 0.7912, 0.7788),
          4: (0.8332, 0.8387, 0.8304)},
         "+0.0697"),
        ("NF-ToN-IoT (8 classes)", (0.20, 0.60),
         [0.4336, 0.2785, 0.4102, 0.4521, 0.5044],
         [0.0053, None, 0.0279, None, 0.0080],
         {0: (0.4290, 0.4393, 0.4326), 2: (0.3975, 0.3910, 0.4422),
          4: (0.4985, 0.5012, 0.5135)},
         "+0.0523"),
    ]
    labels = ["GNN model", "semantic model", "fusion model",
              "feedback model, prototype", "feedback model, trained head"]
    colours = [C_GRAPH, C_SEM, C_FUSE, C_LOOP, C_LOOP]

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.0))
    for ax, (title, ylim, vals, errs, seeds, delta) in zip(axes, panels):
        x = np.arange(5)
        for i, (v, e, c) in enumerate(zip(vals, errs, colours)):
            hollow = (i == 3)
            ax.bar(i, v, width=0.68, zorder=3,
                   color="none" if hollow else c,
                   edgecolor=c, linewidth=0.9,
                   hatch="////" if hollow else None)
            if e is not None:
                ax.errorbar(i, v, yerr=e, fmt="none", ecolor="#333333",
                            elinewidth=0.8, capsize=2.5, zorder=5)
        for i, pts in seeds.items():
            ax.scatter(i + np.array([-0.17, 0.0, 0.17]), pts, s=9,
                       facecolors="none", edgecolors="#333333", linewidths=0.6,
                       zorder=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5,
                           rotation_mode="anchor")
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.65, 4.65)
        ax.set_title(title)
        ax.set_ylabel("pooled out-of-fold macro-F1")
        tidy(ax)

        # the semantic model is deterministic, so it gets a note, not a zero bar
        ax.annotate("deterministic,\nstd 0.0000", xy=(1, vals[1]),
                    xytext=(1, vals[1] + (ylim[1] - ylim[0]) * 0.035),
                    ha="center", va="bottom", fontsize=6.5, color="#555555")

        # the consultant change, drawn between the two feedback model bars
        span = ylim[1] - ylim[0]
        top = min(max(vals[3], vals[4]) + span * 0.075, ylim[1] - span * 0.085)
        ax.annotate("", xy=(4, top), xytext=(3, top),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=C_LOOP))
        ax.text(3.5, top + (ylim[1] - ylim[0]) * 0.015, delta, ha="center",
                va="bottom", fontsize=7.5, color=C_LOOP,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    fig.tight_layout(rect=(0, 0.135, 1, 1))
    fig.text(0.5, 0.012,
             "Error bars are training-seed variance at a fixed fold partition; "
             "fold-partition variance is not measured.\n"
             "Open circles are the three individual seeds. The consultant change is separated on "
             "NF-UNSW-NB15 and not on NF-ToN-IoT;\nthe two feedback model versions also differ in their "
             "selected top-k (31 to 29, and 25 to 16). The two panels have independent y-axes.",
             ha="center", va="bottom", fontsize=6.5, color="#444444")
    save(fig, "fig_ladder")


# --------------------------------------------------------------------------- 4
def fig_imbalance():
    unsw = [("Normal", 311), ("Backdoors", 40), ("DoS", 40), ("Exploits", 40),
            ("Fuzzers", 40), ("Generic", 40), ("Reconnaissance", 40),
            ("Shellcode", 40), ("Worms", 40), ("Analysis", 25)]
    ton = [("Benign", 1746), ("mitm", 157), ("injection", 116), ("ddos", 35),
           ("password", 25), ("backdoor", 16), ("scanning", 13), ("xss", 12),
           ("dos", 4), ("ransomware", 3)]
    excluded = {"dos", "ransomware"}

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.1))
    for ax, data, title, ratio in (
            (axes[0], unsw, "NF-UNSW-NB15, 656 edges", "12.4 : 1"),
            (axes[1], ton, "NF-ToN-IoT, 2,127 edges", "582 : 1")):
        names = [n for n, _ in data]
        counts = [c for _, c in data]
        y = np.arange(len(data))[::-1]
        for yi, (n, c) in zip(y, data):
            rare = c <= 35
            if n in excluded:
                ax.barh(yi, c, height=0.66, color="none", edgecolor=C_GRAPH,
                        linewidth=0.9, linestyle=(0, (2.5, 1.5)), zorder=3)
            else:
                ax.barh(yi, c, height=0.66, color=C_GRAPH, zorder=3)
            if rare:
                ax.axhspan(yi - 0.42, yi + 0.42, color="#d7191c", alpha=0.07,
                           zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(1, 2500)
        ax.set_ylim(-0.7, len(data) - 0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("edges (log scale)")
        ax.set_title(title)
        tidy(ax, grid_axis="x")
        ax.text(0.97, 0.06, ratio, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb",
                          lw=0.6))

    axes[1].annotate("excluded from the metric,\nkept in the graph for\nmessage passing",
                     xy=(4, 1.0), xytext=(18, 2.3), fontsize=6.5,
                     color="#444444", ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888"))

    handles = [Rectangle((0, 0), 1, 1, fc="#d7191c", alpha=0.12, ec="none"),
               Rectangle((0, 0), 1, 1, fc="none", ec=C_GRAPH, lw=0.9,
                         linestyle=(0, (2.5, 1.5)))]
    fig.legend(handles, ["35 edges or fewer (1 class left, 7 right)",
                         "not scored"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.145, 1, 1))
    fig.text(0.5, 0.080,
             "The two graphs differ by roughly a factor of 47 in imbalance severity.",
             ha="center", va="bottom", fontsize=7, color="#444444")
    save(fig, "fig_imbalance")


# --------------------------------------------------------------------------- 6
def fig_mechanism():
    rows_unsw = [
        ("oracle, edge channel", 0.0369, 0.0097, 0.0618, C_ORACLE, True),
        ("trained-head consultant, edge", 0.0007, -0.0183, 0.0201, C_LOOP, False),
        ("prototype consultant, edge", -0.0046, -0.0273, 0.0158, C_NEUT, False),
        ("oracle, attention channel", -0.0007, -0.0211, 0.0180, "#E3B7D2", False),
    ]
    rows_ton = [
        ("oracle, edge channel", 0.1354, 0.0919, 0.1797, C_ORACLE, True),
        ("trained-head consultant, edge", -0.0378, -0.0588, -0.0167, C_LOOP, True),
        ("prototype consultant, edge", 0.0099, -0.0111, 0.0286, C_NEUT, False),
        ("oracle, attention channel", 0.0114, -0.0083, 0.0307, "#E3B7D2", False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.7))
    for ax, rows, title, xlim in ((axes[0], rows_unsw, "NF-UNSW-NB15", (-0.06, 0.09)),
                                  (axes[1], rows_ton, "NF-ToN-IoT", (-0.09, 0.21))):
        ax.axvspan(-0.012, 0.012, color="#bbbbbb", alpha=0.30, zorder=1, lw=0)
        ax.axvline(0, color="#222222", lw=1.0, zorder=2)
        for i, (lab, est, lo, hi, col, sep) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, zorder=4,
                    solid_capstyle="butt")
            for b in (lo, hi):
                ax.plot([b, b], [y - 0.13, y + 0.13], color=col, lw=1.2, zorder=4)
            ax.plot([est], [y], marker="o", ms=4.5, zorder=5,
                    color=col if sep else "white",
                    markerfacecolor=col if sep else "white",
                    markeredgecolor=col, markeredgewidth=1.0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.set_xlabel("paired difference against control")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_axisbelow(True)

    axes[1].set_yticklabels([])
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color="#444444", ms=4.5,
                   label="excludes zero"),
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#444444",
                   ms=4.5, label="includes zero"),
        Rectangle((0, 0), 1, 1, fc="#bbbbbb", alpha=0.30, ec="none",
                  label="run-to-run drift without thread pinning ($\\pm$0.012)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.235, 1, 1))
    fig.text(0.5, 0.085,
             "Output fusion is disabled in every arm, so this measures the injection path "
             "alone. Only the oracle converts the\nchannel; on NF-ToN-IoT the trained-head "
             "consultant is separated in the wrong direction.",
             ha="center", va="bottom", fontsize=6.5, color="#444444")
    save(fig, "fig_mechanism")


# --------------------------------------------------------------------------- 7
def fig_decomposition():
    fig, ax = plt.subplots(figsize=(6.1, 3.1))
    base = 0.3841
    steps = [
        ("fair tuning:\ngraph scale and schedule", 0.2001, "44.5%", "#9ecae1"),
        ("our ten node\ncentralities", 0.1354, "30.1%", C_GRAPH),
        ("architecture", 0.1145, "25.4%", C_LOOP),
    ]
    xs = np.arange(5)
    ax.bar(0, base, width=0.62, color=C_NEUT, zorder=3)
    ax.text(0, base + 0.012, f"{base:.4f}", ha="center", va="bottom", fontsize=7.5)
    run = base
    for i, (lab, inc, share, col) in enumerate(steps, start=1):
        ax.bar(i, inc, bottom=run, width=0.62, color=col, zorder=3)
        ax.plot([i - 0.31 - 0.07, i - 0.31], [run, run], color="#888888", lw=0.6,
                zorder=2)
        ax.plot([i - 0.62, i - 0.31], [run, run], color="#888888", lw=0.6,
                ls=(0, (2, 2)), zorder=2)
        ax.text(i, run + inc + 0.012, f"+{inc:.4f}\n{share}", ha="center",
                va="bottom", fontsize=7.5)
        run += inc
    ax.bar(4, run, width=0.62, color=C_LOOP, zorder=3)
    ax.text(4, run + 0.012, f"{run:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.plot([3.69, 4.31], [0.7793, 0.7793], color="#222222", lw=1.0, zorder=6)
    ax.text(4.36, 0.7793, "fusion model rung\n0.7793", va="center", ha="left", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(["TE-G-SAGE\nas published", "fair tuning:\nscale and schedule",
                        "our ten node\ncentralities", "architecture",
                        "our feedback\nmodel rung"], fontsize=7)
    ax.set_ylim(0.30, 0.92)
    ax.set_xlim(-0.6, 5.35)
    ax.set_ylabel("pooled out-of-fold macro-F1")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.text(0.5, 0.018,
             "Point estimates only; every level is a three-seed mean and no step carries "
             "an interval or a separation claim.",
             ha="center", va="bottom", fontsize=6.5, color="#444444")
    save(fig, "fig_decomposition")


# --------------------------------------------------------------------------- 8
def fig_perclass():
    unsw = [("Normal", 311, 0.0390, 0.0414), ("Analysis", 25, 0.0309, 0.1869),
            ("Backdoors", 40, 0.1442, 0.2090), ("DoS", 40, 0.0978, 0.1594),
            ("Exploits", 40, -0.0003, 0.0271), ("Fuzzers", 40, 0.0419, 0.1542),
            ("Generic", 40, 0.1490, 0.0308), ("Reconnaissance", 40, -0.0185, -0.0379),
            ("Shellcode", 40, 0.0531, 0.1290), ("Worms", 40, 0.0107, 0.0040)]
    ton = [("Benign", 1746, 0.0558, 0.0234), ("backdoor", 16, 0.2027, 0.0834),
           ("ddos", 35, 0.0494, 0.1064), ("injection", 116, 0.1433, 0.0842),
           ("mitm", 157, 0.1484, -0.0273), ("password", 25, -0.0850, 0.1699),
           ("scanning", 13, 0.0335, 0.2159), ("xss", 12, 0.1768, -0.1116)]

    rows, labels = [[np.nan, np.nan]], [""]
    for name, n, a, b in unsw:
        rows.append([a, b]); labels.append(f"{name}  ({n})")
    rows.append([np.nan, np.nan]); labels.append("")
    for name, n, a, b in ton:
        rows.append([a, b]); labels.append(f"{name}  ({n})")
    M = np.array(rows)

    fig, ax = plt.subplots(figsize=(6.1, 4.3))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.22, vmax=0.22, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"feedback {MINUS} fusion", f"feedback {MINUS} GNN"])
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    for i in range(M.shape[0]):
        for j in range(2):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"${v:+.4f}$", ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(v) > 0.15 else "#222222")
            if v < 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec="#111111", lw=1.0))
    ax.text(-0.45, 0, "NF-UNSW-NB15", va="center", ha="left", fontsize=8.5)
    ax.text(-0.45, 11, "NF-ToN-IoT", va="center", ha="left", fontsize=8.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label("difference in per-class F1", fontsize=7.5)

    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.text(0.5, 0.018,
             "Three-seed means. Class sizes in brackets. A black outline marks a loss. "
             "No per-class difference carries an interval,\nand on classes of 12 to 40 "
             "edges a single edge moves F1 substantially.",
             ha="center", va="bottom", fontsize=6.5, color="#444444")
    save(fig, "fig_perclass")


# --------------------------------------------------------------------------- 9
def fig_comparisons():
    # (label, two-level est/lo/hi, seed-matched est/lo/hi)
    unsw = [
        ("head", f"feedback {MINUS} semantic", (0.0989, 0.0721, 0.1267), (0.0991, 0.0743, 0.1253)),
        ("head", f"feedback {MINUS} GNN", (0.0908, 0.0412, 0.1407), (0.0907, 0.0571, 0.1250)),
        ("head", f"feedback {MINUS} fusion", (0.0546, 0.0228, 0.0880), (0.0551, 0.0331, 0.0779)),
        ("proto", f"feedback {MINUS} GNN", (0.0206, -0.0203, 0.0625), (0.0207, -0.0001, 0.0421)),
        ("proto", f"feedback {MINUS} fusion", (-0.0156, -0.0552, 0.0268), (-0.0150, -0.0425, 0.0126)),
        ("indep", f"fusion {MINUS} GNN", (0.0362, -0.0096, 0.0822), (0.0356, 0.0038, 0.0678)),
    ]
    ton = [
        ("head", f"feedback {MINUS} semantic", (0.2231, 0.1703, 0.2771), (0.2228, 0.1732, 0.2713)),
        ("head", f"feedback {MINUS} GNN", (0.0694, 0.0129, 0.1288), (0.0691, 0.0171, 0.1200)),
        ("head", f"feedback {MINUS} fusion", (0.0932, 0.0180, 0.1667), (0.0940, 0.0406, 0.1480)),
        ("proto", f"feedback {MINUS} GNN", (0.0191, -0.0180, 0.0596), (0.0182, -0.0050, 0.0418)),
        ("proto", f"feedback {MINUS} fusion", (0.0428, -0.0316, 0.1105), (0.0431, -0.0059, 0.0936)),
        ("indep", f"fusion {MINUS} GNN", (-0.0238, -0.0921, 0.0607), (-0.0249, -0.0697, 0.0226)),
    ]
    tone = {"head": {"loop": C_LOOP, "AGAF": C_FUSE},
            "proto": {"loop": "#F0A882", "AGAF": "#8ED9C1"},
            "indep": {"loop": C_FUSE, "AGAF": C_FUSE}}

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.6))
    for ax, rows, title, xlim in ((axes[0], unsw, "NF-UNSW-NB15", (-0.08, 0.155)),
                                  (axes[1], ton, "NF-ToN-IoT", (-0.11, 0.30))):
        ax.axvline(0, color="#222222", lw=1.0, zorder=2)
        n = len(rows)
        for i, (grp, lab, two, sm) in enumerate(rows):
            y = n - 1 - i
            col = tone[grp]["AGAF"] if grp == "indep" else tone[grp]["loop"]
            light = "#bbbbbb" if grp == "indep" else None
            for (est, lo, hi), off, mk, lw in ((two, 0.16, "o", 1.2),
                                               (sm, -0.16, "s", 1.0)):
                cc = col if off > 0 else matplotlib.colors.to_rgba(col, 0.55)
                ax.plot([lo, hi], [y + off] * 2, color=cc, lw=lw, zorder=4)
                for b in (lo, hi):
                    ax.plot([b, b], [y + off - 0.10, y + off + 0.10], color=cc,
                            lw=lw, zorder=4)
                filled = lo > 0 or hi < 0
                ax.plot([est], [y + off], marker=mk, ms=4.0 if mk == "o" else 3.4,
                        mfc=cc if filled else "white", mec=cc, mew=1.0, zorder=5)
        # group separators
        for yline in (2.5, 0.5):
            ax.axhline(yline, color="#dddddd", lw=0.6, zorder=1)
        for yh, name in ((5.58, "trained-head consultant"),
                         (2.58, "prototype consultant"),
                         (0.58, "consultant-independent")):
            ax.text(0.012, yh, name, transform=ax.get_yaxis_transform(),
                    fontsize=6.8, color="#555555", va="bottom", ha="left")
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[1] for r in rows][::-1])
        ax.set_ylim(-0.6, n - 0.05)
        ax.set_xlim(*xlim)
        ax.set_title(title)
        ax.set_xlabel("difference in pooled out-of-fold macro-F1")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[1].set_yticklabels([])
    axes[0].text(0.0822, 0.16, "*", fontsize=9, color="#444444", va="center",
                 ha="left")

    handles = [
        plt.Line2D([], [], marker="o", color="#555555", ms=4.0, lw=1.2,
                   label="two-level (edges and training seed)"),
        plt.Line2D([], [], marker="s", color="#aaaaaa", ms=3.4, lw=1.0,
                   label="seed-matched (edges only)"),
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555",
                   ms=4.0, label="interval includes zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.175, 1, 1))
    fig.text(0.5, 0.075,
             "* On NF-UNSW-NB15, the fusion model minus the GNN model is separated under the "
             "seed-matched procedure and not once training-seed\nvariance is admitted; the "
             "two-level result decides. There is no feedback-versus-semantic-model comparison "
             "for the prototype consultant.",
             ha="center", va="bottom", fontsize=6.5, color="#444444")
    save(fig, "fig_comparisons")


FIGS = {"ladder": fig_ladder, "imbalance": fig_imbalance,
        "mechanism": fig_mechanism, "decomposition": fig_decomposition,
        "perclass": fig_perclass, "comparisons": fig_comparisons}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGS)
    for name in wanted:
        FIGS[name]()
