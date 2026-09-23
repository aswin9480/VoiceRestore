import json
from pathlib import Path


def write_report(folder: Path, report: dict) -> None:
    (folder / "processing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "VoiceRestore Studio Processing Report",
        "=" * 40,
        f"Status: {report['status']}",
        f"Input: {report['input']['filename']}",
        f"Original duration: {report['input']['duration']:.3f} s",
        f"Processing duration: {report['processing_seconds']:.2f} s",
        "",
        "Pipeline:",
    ]
    for stage in report["pipeline"]:
        lines.append(f"  {stage['stage']}: {stage['status']}")
        for key in ["model", "version", "device", "settings"]:
            if key in stage:
                lines.append(f"    {key}: {stage[key]}")
    lines += [
        "",
        "All requested settings:",
        json.dumps(report["settings"], indent=2),
        "",
        "Measured final loudness / peaks:",
        json.dumps(report.get("measurements", {}), indent=2),
        "",
        "Outputs:",
        *report.get("outputs", []),
    ]
    if report.get("error"):
        lines += ["", "Error:", report["error"]]
    (folder / "processing_report.txt").write_text("\n".join(lines), encoding="utf-8")
