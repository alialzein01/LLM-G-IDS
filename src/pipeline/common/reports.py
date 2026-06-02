from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.common.datasets import DatasetConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def report_dir(config: DatasetConfig, phase: str) -> Path:
    return Path("data") / config.key / "processed" / "reports" / phase


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_markdown(path: str | Path, title: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append(f"- Dataset: `{payload.get('dataset')}`")
    lines.append(f"- Status: `{payload.get('status')}`")
    lines.append(f"- Generated at: `{payload.get('generated_at')}`")
    lines.append("")

    checks = payload.get("checks", [])
    if checks:
        lines.append("## Checks")
        for check in checks:
            status = "PASS" if check.get("passed") else "FAIL"
            detail = check.get("detail")
            suffix = f" - {detail}" if detail else ""
            lines.append(f"- `{status}` {check.get('name')}{suffix}")
        lines.append("")

    metrics = payload.get("metrics")
    if metrics:
        lines.append("## Metrics")
        for key, value in metrics.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    artifacts = payload.get("artifacts", {})
    if artifacts:
        lines.append("## Artifacts")
        for key, value in artifacts.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    if payload.get("notes"):
        lines.append("## Notes")
        for note in payload["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    out.write_text("\n".join(lines) + "\n")


def approval_manifest(
    *,
    phase: str,
    dataset: str,
    status: str,
    checks: list[dict[str, Any]],
    artifacts: dict[str, str],
    metrics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "dataset": dataset,
        "status": status,
        "generated_at": utc_now_iso(),
        "manual_approval": {
            "approved": False,
            "approved_by": None,
            "approved_at": None,
            "notes": "",
        },
        "checks": checks,
        "metrics": metrics or {},
        "artifacts": artifacts,
        "notes": notes or [],
    }


def check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def finalize_report(
    config: DatasetConfig,
    phase: str,
    title: str,
    checks: list[dict[str, Any]],
    artifacts: dict[str, str],
    metrics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    status = "passed" if all(item["passed"] for item in checks) else "failed"
    payload = approval_manifest(
        phase=phase,
        dataset=config.key,
        status=status,
        checks=checks,
        artifacts=artifacts,
        metrics=metrics,
        notes=notes,
    )
    out = report_dir(config, phase)
    write_json(out / "validation_report.json", payload)
    write_markdown(out / "validation_report.md", title, payload)
    write_json(out / "approval_manifest.json", payload)
    return payload
