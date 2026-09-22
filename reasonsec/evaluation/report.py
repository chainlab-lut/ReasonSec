from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from reasonsec.utils.io import ensure_directory, write_json


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_line = "| " + " | ".join(str(header) for header in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join("" if value is None else str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_csv(path: str | Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    target = Path(path).expanduser()
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return target


def format_percentage(value: float, digits: int = 1) -> str:
    return f"{value * 100.0:.{digits}f}"


def format_mean_deviation(mean: float, deviation: float, digits: int = 1) -> str:
    return f"{mean * 100.0:.{digits}f} ± {deviation * 100.0:.{digits}f}"


def per_category_table(
    methods: Sequence[str],
    per_method_categories: Mapping[str, Mapping[str, Mapping[str, float]]],
    categories: Sequence[str],
) -> tuple[list[str], list[list[Any]]]:
    headers = ["CWE", "N", *methods]
    rows: list[list[Any]] = []
    for category in categories:
        totals = [
            int(per_method_categories[method][category]["total"])
            for method in methods
            if category in per_method_categories.get(method, {})
        ]
        row: list[Any] = [category, totals[0] if totals else 0]
        for method in methods:
            entry = per_method_categories.get(method, {}).get(category)
            if entry is None:
                row.append("")
            else:
                row.append(f"{int(entry['avoided'])}/{int(entry['total'])} ({format_percentage(entry['rate'])})")
        rows.append(row)
    return headers, rows


def method_summary_table(summaries: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Method",
        "Security rate (%)",
        "pass@1 (%)",
        "Knowledge accuracy (%)",
        "Secondary benchmark security rate (%)",
    ]
    rows: list[list[Any]] = []
    for summary in summaries:
        rows.append(
            [
                summary.get("method", ""),
                summary.get("security_rate_formatted", ""),
                summary.get("pass_at_1_formatted", ""),
                summary.get("knowledge_formatted", ""),
                summary.get("secondary_security_formatted", ""),
            ]
        )
    return headers, rows


def write_report(path: str | Path, title: str, sections: Sequence[tuple[str, str]]) -> Path:
    target = Path(path).expanduser()
    ensure_directory(target.parent)
    lines = [f"# {title}", ""]
    for heading, content in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(content)
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def save_results(directory: str | Path, name: str, payload: Any) -> Path:
    return write_json(Path(directory).expanduser() / f"{name}.json", payload)
