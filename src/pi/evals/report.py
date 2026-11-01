"""Markdown reports for one run, and comparison tables across runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OVERALL_ROWS: list[tuple[str, str, str]] = [
    ("success_rate", "Success rate", "pct"),
    ("numeric_accuracy", "Numerical accuracy", "pct"),
    ("mention_accuracy", "Expected mentions found", "pct"),
    ("tool_precision", "Tool precision", "pct"),
    ("tool_recall", "Tool recall", "pct"),
    ("grounding_rate", "Grounding rate", "pct"),
    ("answers_with_hallucinations", "Answers with ungrounded numbers", "int"),
    ("hallucinated_numbers", "Ungrounded numbers (total)", "int"),
    ("access_violations", "Access violations", "int"),
    ("refusal_accuracy", "Correct refusals", "pct"),
    ("failure_rate", "Failures (timeouts, loops, errors)", "pct"),
    ("latency_p50_s", "Latency p50 (s)", "f1"),
    ("latency_p95_s", "Latency p95 (s)", "f1"),
    ("mean_tokens", "Tokens per question", "int"),
    ("mean_iterations", "Model calls per question", "f1"),
    ("cost_usd", "Cost (USD)", "f2"),
]
CATEGORY_COLS: list[tuple[str, str, str]] = [
    ("items", "Items", "int"),
    ("success_rate", "Success", "pct"),
    ("numeric_accuracy", "Numbers", "pct"),
    ("tool_recall", "Tool recall", "pct"),
    ("grounding_rate", "Grounding", "pct"),
    ("latency_p50_s", "p50 (s)", "f1"),
]


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "–"
    if kind == "pct":
        return f"{100 * float(value):.0f}%"
    if kind == "int":
        return f"{float(value):,.0f}"
    if kind == "f1":
        return f"{float(value):.1f}"
    return f"{float(value):.2f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def run_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    overall = summary["overall"]
    parts = [f"# Evaluation report: `{summary.get('model', '?')}`", ""]
    parts.append(f"{overall['items']} questions.\n")
    parts.append(
        _table(
            ["Metric", "Value"],
            [[label, fmt(overall.get(key), kind)] for key, label, kind in OVERALL_ROWS],
        )
    )
    parts += ["", "## By category", ""]
    parts.append(
        _table(
            ["Category"] + [label for _, label, _ in CATEGORY_COLS],
            [
                [cat] + [fmt(agg.get(key), kind) for key, _, kind in CATEGORY_COLS]
                for cat, agg in summary["by_category"].items()
            ],
        )
    )
    failures = [r for r in rows if not r["success"]]
    if failures:
        parts += ["", "## Unsuccessful items", ""]
        for r in sorted(failures, key=lambda r: r["id"]):
            answer = " ".join(str(r["answer"]).split())[:300]
            parts.append(f"- **{r['id']}** ({r['status']}): {r['question']}")
            parts.append(
                f"  - tools: {', '.join(r['tools_called']) or 'none'}; "
                f"expected: {', '.join(r['expected_tools']) or 'none'}"
            )
            if r["ungrounded"]:
                parts.append(f"  - ungrounded: {', '.join(r['ungrounded'])}")
            parts.append(f"  - answer: {answer}")
    return "\n".join(parts) + "\n"


def write_report(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    (out / "report.md").write_text(run_report(summary, rows), encoding="utf-8")


def compare_runs(run_dirs: list[Path]) -> str:
    summaries = [json.loads((d / "summary.json").read_text()) for d in run_dirs]
    names = [
        str(s.get("model", d.name)).split("/")[-1] for s, d in zip(summaries, run_dirs, strict=True)
    ]
    parts = ["## Overall", ""]
    parts.append(
        _table(
            ["Metric", *names],
            [
                [label] + [fmt(s["overall"].get(key), kind) for s in summaries]
                for key, label, kind in OVERALL_ROWS
            ],
        )
    )
    categories = sorted({c for s in summaries for c in s["by_category"]})
    for key, label, kind in [
        ("success_rate", "Success rate", "pct"),
        ("numeric_accuracy", "Numerical accuracy", "pct"),
        ("latency_p50_s", "Latency p50 (s)", "f1"),
    ]:
        parts += ["", f"## {label} by category", ""]
        parts.append(
            _table(
                ["Category", *names],
                [
                    [c] + [fmt(s["by_category"].get(c, {}).get(key), kind) for s in summaries]
                    for c in categories
                ],
            )
        )
    return "\n".join(parts) + "\n"
