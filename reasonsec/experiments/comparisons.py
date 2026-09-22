from __future__ import annotations

from typing import Any, Sequence

from reasonsec.config import Config
from reasonsec.evaluation.report import (
    format_mean_deviation,
    format_percentage,
    per_category_table,
    render_markdown_table,
    write_csv,
    write_report,
)
from reasonsec.evaluation.security import avoidance_identifiers, paired_outcomes
from reasonsec.evaluation.statistics import (
    mcnemar_test,
    minimum_detectable_difference,
    paired_t_test,
)
from reasonsec.experiments.runner import ExperimentRunner, MethodEvaluation
from reasonsec.utils.io import write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _statistical_comparison(
    runner: ExperimentRunner,
    reference: MethodEvaluation,
    other: MethodEvaluation,
    config: Config,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {"reference": reference.method, "method": other.method}
    shared_seeds = sorted(set(reference.security) & set(other.security))
    if shared_seeds:
        seed = shared_seeds[0]
        left, right = paired_outcomes(reference.security[seed], other.security[seed])
        if left and right:
            test = mcnemar_test(
                left,
                right,
                exact=config.require_bool("evaluation.statistics.mcnemar_exact"),
                continuity_correction=config.require_bool("evaluation.statistics.continuity_correction"),
            )
            comparison["security"] = test.as_dict()
            comparison["minimum_detectable_difference"] = minimum_detectable_difference(
                sample_size=len(left),
                baseline_rate=sum(right) / len(right) if right else 0.0,
                power=config.require_float("evaluation.statistics.power"),
                alpha=config.require_float("evaluation.statistics.alpha"),
            )
    reference_pass = reference.pass_at_one_values()
    other_pass = other.pass_at_one_values()
    if len(reference_pass) == len(other_pass) and len(reference_pass) > 1:
        comparison["pass_at_1"] = paired_t_test(reference_pass, other_pass).as_dict()
    return comparison


def run_method_comparison(
    runner: ExperimentRunner,
    name: str,
    methods: Sequence[str],
    reference_method: str,
    include_per_category: bool,
) -> dict[str, Any]:
    config = runner.config
    runner.prepare(methods)
    evaluations = {method: runner.evaluate_method(method) for method in methods}
    summaries: list[dict[str, Any]] = []
    for method in methods:
        aggregate = evaluations[method].aggregate()
        aggregate["security_rate_formatted"] = format_mean_deviation(
            aggregate["security_rate_mean"], aggregate["security_rate_standard_deviation"]
        )
        aggregate["pass_at_1_formatted"] = format_mean_deviation(
            aggregate["pass_at_1_mean"], aggregate["pass_at_1_standard_deviation"]
        )
        aggregate["knowledge_formatted"] = format_percentage(aggregate["knowledge_accuracy_mean"])
        aggregate["secondary_security_formatted"] = format_percentage(aggregate["secondary_security_rate_mean"])
        summaries.append(aggregate)

    reference = evaluations[reference_method]
    comparisons = [
        _statistical_comparison(runner, reference, evaluations[method], config)
        for method in methods
        if method != reference_method
    ]

    headers = [
        "Method",
        "Security rate (%)",
        "pass@1 (%)",
        "Knowledge accuracy (%)",
        "Secondary benchmark security rate (%)",
    ]
    rows = [
        [
            summary["method"],
            summary["security_rate_formatted"],
            summary["pass_at_1_formatted"],
            summary["knowledge_formatted"],
            summary["secondary_security_formatted"],
        ]
        for summary in summaries
    ]
    directory = runner.root / "experiments" / name
    write_csv(directory / "summary.csv", headers, rows)
    sections = [("Method comparison", render_markdown_table(headers, rows))]

    payload: dict[str, Any] = {
        "name": name,
        "methods": list(methods),
        "reference_method": reference_method,
        "summaries": summaries,
        "comparisons": comparisons,
    }

    if include_per_category:
        per_method_categories = {method: evaluations[method].per_category() for method in methods}
        categories = sorted(
            {cwe for values in per_method_categories.values() for cwe in values},
            key=lambda item: int(item.split("-")[1]),
        )
        category_headers, category_rows = per_category_table(list(methods), per_method_categories, categories)
        write_csv(directory / "per_category.csv", category_headers, category_rows)
        sections.append(("Per-category avoidance", render_markdown_table(category_headers, category_rows)))
        payload["per_category"] = per_method_categories

    write_json(directory / "results.json", payload)
    write_report(directory / "report.md", f"Experiment: {name}", sections)
    LOGGER.info("completed experiment '%s'", name)
    return payload


def run_avoidance_overlap(runner: ExperimentRunner, name: str, methods: Sequence[str]) -> dict[str, Any]:
    evaluations = {method: runner.evaluate_method(method) for method in methods}
    seeds = sorted(set.intersection(*[set(evaluation.security) for evaluation in evaluations.values()]))
    if not seeds:
        return {}
    seed = seeds[0]
    sets = {method: avoidance_identifiers(evaluations[method].security[seed]) for method in methods}
    payload: dict[str, Any] = {"seed": seed, "sizes": {method: len(values) for method, values in sets.items()}}
    regions: dict[str, int] = {}
    method_list = list(methods)
    for mask in range(1, 1 << len(method_list)):
        included = [method_list[index] for index in range(len(method_list)) if mask & (1 << index)]
        excluded = [method for method in method_list if method not in included]
        region = set.intersection(*[sets[method] for method in included])
        for method in excluded:
            region = region - sets[method]
        regions["+".join(included)] = len(region)
    payload["regions"] = regions
    directory = runner.root / "experiments" / name
    write_json(directory / "overlap.json", payload)
    return payload
