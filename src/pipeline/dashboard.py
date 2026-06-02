from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from src.pipeline.common.datasets import DATASETS, DatasetConfig, get_dataset_config


OUTPUT_DIR = Path("data/dashboard")
OUTPUT_HTML = OUTPUT_DIR / "index.html"


def _load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _esc(value: object) -> str:
    return html.escape(str(value))


def _summary(config: DatasetConfig) -> dict[str, Any]:
    reports_root = Path("data") / config.key / "processed" / "reports"
    phase_reports = {
        phase: _load_json(reports_root / phase / "validation_report.json")
        for phase in ("phase1", "phase2", "phase3", "phase4")
    }
    model_summaries = {
        "GATv2 Only": _load_json(Path(config.gnn_output_dir) / "benchmark_summary.json"),
        "CySecBERT Only": _load_json(
            Path(config.baseline_output_dir) / "llm_embedding" / "benchmark_summary.json"
        ),
        "Fusion Model": _load_json(Path(config.fusion_output_dir) / "benchmark_summary.json"),
    }
    confusion = _load_json(reports_root / "phase4" / "confusion_matrix.json")
    significance = _load_json(reports_root / "phase4" / "significance_report.json")
    fusion_metrics = _load_json(Path(config.fusion_output_dir) / "metrics.json")
    return {
        "key": config.key,
        "display_name": config.display_name,
        "phase_reports": phase_reports,
        "model_summaries": model_summaries,
        "confusion": confusion,
        "significance": significance,
        "fusion_metrics": fusion_metrics,
        "exports": {
            "phase1_report": f"../{config.key}/processed/reports/phase1/validation_report.json",
            "phase2_report": f"../{config.key}/processed/reports/phase2/validation_report.json",
            "phase3_report": f"../{config.key}/processed/reports/phase3/validation_report.json",
            "phase4_report": f"../{config.key}/processed/reports/phase4/validation_report.json",
            "aggregated_edges": f"../{config.key}/processed/step1/aggregated_edges.csv",
            "kg_triples": f"../{config.key}/processed/step2/kg_triples.csv",
        },
    }


def _metric_table(model_summaries: dict[str, dict[str, Any] | None]) -> str:
    rows = []
    for model, summary in model_summaries.items():
        if summary is None:
            rows.append(f"<tr><td>{_esc(model)}</td><td colspan='4'>Missing</td></tr>")
            continue
        rows.append(
            "<tr>"
            f"<td>{_esc(model)}</td>"
            f"<td>{summary.get('pooled_cv_accuracy', 0):.4f}</td>"
            f"<td>{summary.get('pooled_cv_macro_f1', 0):.4f}</td>"
            f"<td>{summary.get('pooled_cv_weighted_f1', 0):.4f}</td>"
            f"<td>{summary.get('cv_test_macro_f1_mean', 0):.4f} ± {summary.get('cv_test_macro_f1_std', 0):.4f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th>"
        "<th>Weighted-F1</th><th>Fold Macro-F1</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _phase_cards(phase_reports: dict[str, dict[str, Any] | None]) -> str:
    cards = []
    for phase, report in phase_reports.items():
        if report is None:
            cards.append(f"<section class='card'><h3>{phase}</h3><p class='fail'>Missing report</p></section>")
            continue
        status = report.get("status", "unknown")
        cls = "pass" if status == "passed" else "fail"
        checks = report.get("checks", [])
        passed = sum(1 for item in checks if item.get("passed"))
        cards.append(
            "<section class='card'>"
            f"<h3>{_esc(phase)}</h3>"
            f"<p class='{cls}'>{_esc(status)}</p>"
            f"<p>{passed}/{len(checks)} checks passed</p>"
            "</section>"
        )
    return "<div class='cards'>" + "".join(cards) + "</div>"


def _bars(model_summaries: dict[str, dict[str, Any] | None]) -> str:
    bars = []
    for model, summary in model_summaries.items():
        value = float(summary.get("pooled_cv_macro_f1", 0.0)) if summary else 0.0
        bars.append(
            "<div class='bar-row'>"
            f"<span>{_esc(model)}</span>"
            f"<div class='bar'><i style='width:{max(0.0, min(1.0, value)) * 100:.1f}%'></i></div>"
            f"<strong>{value:.4f}</strong>"
            "</div>"
        )
    return "".join(bars)


def _confusion(confusion: dict[str, Any] | None) -> str:
    if not confusion or not confusion.get("matrix"):
        return "<p class='muted'>Confusion matrix is unavailable until Phase 4 validation runs.</p>"
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    header = "<tr><th>Actual \\ Pred</th>" + "".join(f"<th>{_esc(label)}</th>" for label in labels) + "</tr>"
    rows = []
    for label, values in zip(labels, matrix):
        cells = "".join(f"<td>{int(v)}</td>" for v in values)
        rows.append(f"<tr><th>{_esc(label)}</th>{cells}</tr>")
    return "<div class='scroll'><table class='matrix'>" + header + "".join(rows) + "</table></div>"


def _attention(fusion_metrics: dict[str, Any] | None, significance: dict[str, Any] | None) -> str:
    parts = []
    diagnostics = (fusion_metrics or {}).get("fusion_diagnostics_summary", [])
    if diagnostics:
        rows = []
        for row in diagnostics:
            rows.append(
                "<tr>"
                f"<td>{_esc(row['class_name'])}</td>"
                f"<td>{row['support']}</td>"
                f"<td>{row.get('gate_gnn_mean', 0):.4f}</td>"
                f"<td>{row.get('gate_llm_mean', 0):.4f}</td>"
                f"<td>{row.get('feature_attention_entropy_mean', 0):.4f}</td>"
                "</tr>"
            )
        parts.append(
            "<table><thead><tr><th>Class</th><th>Support</th><th>Gate GNN</th>"
            "<th>Gate LLM</th><th>Feature Entropy</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    else:
        parts.append("<p class='muted'>Attention diagnostics are unavailable.</p>")
    if significance:
        parts.append("<pre>" + _esc(json.dumps(significance, indent=2)) + "</pre>")
    return "".join(parts)


def _dataset_section(summary: dict[str, Any], active: bool) -> str:
    display = "" if active else " hidden"
    exports = "".join(
        f"<a href='{_esc(path)}' download>{_esc(name)}</a>"
        for name, path in summary["exports"].items()
    )
    phase1_metrics = (summary["phase_reports"].get("phase1") or {}).get("metrics", {})
    dataset_stats = json.dumps(
        {
            "nodes": phase1_metrics.get("graph_nodes"),
            "edges": phase1_metrics.get("graph_edges"),
            "connectivity": phase1_metrics.get("connectivity"),
        },
        indent=2,
    )
    return (
        f"<main id='panel-{_esc(summary['key'])}' class='dataset{display}'>"
        f"<h2>{_esc(summary['display_name'])}</h2>"
        "<h3>Phase Status</h3>"
        f"{_phase_cards(summary['phase_reports'])}"
        "<h3>Dataset Statistics</h3>"
        f"<pre>{_esc(dataset_stats)}</pre>"
        "<h3>Model Results</h3>"
        f"{_metric_table(summary['model_summaries'])}"
        "<h3>F1 Comparison</h3>"
        f"{_bars(summary['model_summaries'])}"
        "<h3>Confusion Matrix</h3>"
        f"{_confusion(summary['confusion'])}"
        "<h3>Explainability</h3>"
        f"{_attention(summary['fusion_metrics'], summary['significance'])}"
        "<h3>Exports</h3>"
        f"<div class='exports'>{exports}</div>"
        "</main>"
    )


def generate(output_path: Path = OUTPUT_HTML) -> Path:
    summaries = [_summary(get_dataset_config(key)) for key in sorted(DATASETS)]
    tabs = "".join(
        f"<button data-target='panel-{_esc(item['key'])}'>{_esc(item['display_name'])}</button>"
        for item in summaries
    )
    sections = "".join(_dataset_section(item, i == 0) for i, item in enumerate(summaries))
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM-G-IDS Dashboard</title>
<style>
:root {{ color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ margin: 0; background: #f5f7fb; color: #172033; }}
header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d9e0ea; padding: 16px 24px; }}
h1 {{ margin: 0 0 12px; font-size: 24px; }}
h2 {{ margin-top: 0; }}
h3 {{ margin: 28px 0 10px; font-size: 16px; }}
button {{ border: 1px solid #b9c5d6; background: #ffffff; border-radius: 6px; padding: 8px 12px; margin-right: 8px; cursor: pointer; }}
button.active {{ background: #12355b; color: #ffffff; border-color: #12355b; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
.hidden {{ display: none; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.card {{ background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; padding: 14px; }}
.card h3 {{ margin: 0 0 8px; }}
.pass {{ color: #087443; font-weight: 700; }}
.fail {{ color: #b42318; font-weight: 700; }}
.muted {{ color: #667085; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d9e0ea; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e6ebf2; text-align: left; font-size: 13px; }}
th {{ background: #eef3f8; }}
pre {{ background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; padding: 12px; overflow: auto; }}
.bar-row {{ display: grid; grid-template-columns: 170px 1fr 80px; gap: 12px; align-items: center; margin: 10px 0; }}
.bar {{ height: 18px; background: #dce4ee; border-radius: 5px; overflow: hidden; }}
.bar i {{ display: block; height: 100%; background: #2e6f9e; }}
.scroll {{ overflow: auto; border: 1px solid #d9e0ea; }}
.matrix {{ min-width: 900px; border: 0; }}
.exports {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.exports a {{ color: #12355b; background: #ffffff; border: 1px solid #b9c5d6; border-radius: 6px; padding: 7px 10px; text-decoration: none; }}
</style>
</head>
<body>
<header>
<h1>LLM-G-IDS Dashboard</h1>
<nav>{tabs}</nav>
</header>
{sections}
<script>
const buttons = Array.from(document.querySelectorAll("button[data-target]"));
function activate(target) {{
  buttons.forEach(button => button.classList.toggle("active", button.dataset.target === target));
  document.querySelectorAll("main.dataset").forEach(panel => panel.classList.toggle("hidden", panel.id !== target));
}}
buttons.forEach(button => button.addEventListener("click", () => activate(button.dataset.target)));
if (buttons[0]) activate(buttons[0].dataset.target);
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text)
    return output_path


def serve(path: Path, port: int) -> None:
    root = path.parent.parent if path.parent.name == "dashboard" else path.parent
    served_path = "/" + path.relative_to(root).as_posix()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(root), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving dashboard at http://127.0.0.1:{port}{served_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or serve the LLM-G-IDS dashboard.")
    parser.add_argument("--output", default=str(OUTPUT_HTML))
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = generate(Path(args.output))
    print(f"Dashboard written to {out.resolve()}")
    if args.serve:
        serve(out, args.port)
        sys.exit(0)
