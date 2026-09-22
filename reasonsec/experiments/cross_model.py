from __future__ import annotations

from typing import Any, Mapping

from reasonsec.config import Config
from reasonsec.evaluation.report import format_mean_deviation, render_markdown_table, write_csv, write_report
from reasonsec.experiments.runner import ExperimentRunner
from reasonsec.utils.io import write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def run_cross_model(config: Config, name: str) -> dict[str, Any]:
    section = config.require_section(f"experiments.{name}")
    model_entries = section.require_list("models")
    methods = [str(item) for item in section.require_list("methods")]
    reference_method = section.require_str("reference_method")
    baseline_method = section.require_str("baseline_method")
    rows: list[list[Any]] = []
    payload: dict[str, Any] = {"name": name, "models": []}

    for entry in model_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("each entry of the cross-model experiment must be a mapping")
        label = str(entry["label"])
        overrides = {str(key): value for key, value in entry.get("overrides", {}).items()}
        overrides.setdefault("run.output_directory", str(config.require_path("run.output_directory", must_exist=False) / "models" / label))
        model_config = config.derive(overrides)
        runner = ExperimentRunner(model_config)
        runner.prepare(methods)
        evaluations = {method: runner.evaluate_method(method) for method in methods}
        aggregates = {method: evaluations[method].aggregate() for method in methods}
        baseline_rate = aggregates[baseline_method]["security_rate_mean"]
        proposed_rate = aggregates[reference_method]["security_rate_mean"]
        record = {
            "label": label,
            "parameters": entry.get("parameters"),
            "aggregates": aggregates,
            "security_rate_difference": proposed_rate - baseline_rate,
        }
        payload["models"].append(record)
        rows.append(
            [
                label,
                entry.get("parameters", ""),
                f"{baseline_rate * 100.0:.1f}",
                format_mean_deviation(
                    aggregates[reference_method]["security_rate_mean"],
                    aggregates[reference_method]["security_rate_standard_deviation"],
                ),
                f"{(proposed_rate - baseline_rate) * 100.0:+.1f}",
                format_mean_deviation(
                    aggregates[baseline_method]["pass_at_1_mean"],
                    aggregates[baseline_method]["pass_at_1_standard_deviation"],
                ),
                format_mean_deviation(
                    aggregates[reference_method]["pass_at_1_mean"],
                    aggregates[reference_method]["pass_at_1_standard_deviation"],
                ),
            ]
        )

    headers = [
        "Model",
        "Parameters",
        "Baseline security rate (%)",
        "Proposed security rate (%)",
        "Difference (pp)",
        "Baseline pass@1 (%)",
        "Proposed pass@1 (%)",
    ]
    directory = config.require_path("run.output_directory", must_exist=False) / "experiments" / name
    write_csv(directory / "cross_model.csv", headers, rows)
    write_json(directory / "results.json", payload)
    write_report(directory / "report.md", f"Experiment: {name}", [("Cross-model generalisation", render_markdown_table(headers, rows))])
    LOGGER.info("completed cross-model experiment '%s' over %d models", name, len(rows))
    return payload
