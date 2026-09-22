from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reasonsec.config import Config
from reasonsec.utils.io import ensure_directory, read_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _results_path(config: Config, name: str) -> Path:
    return Path(str(config.require("run.output_directory"))).expanduser() / "experiments" / name / "results.json"


def _figure_path(config: Config, name: str, suffix: str) -> Path:
    directory = ensure_directory(
        Path(str(config.require("run.output_directory"))).expanduser() / "figures"
    )
    return directory / f"{name}_{suffix}.png"


def _method_comparison_figures(config: Config, name: str, payload: dict[str, Any]) -> list[Path]:
    summaries = payload.get("summaries", [])
    if not summaries:
        return []
    methods = [summary["method"] for summary in summaries]
    security = [summary["security_rate_mean"] * 100.0 for summary in summaries]
    security_error = [summary["security_rate_standard_deviation"] * 100.0 for summary in summaries]
    functional = [summary["pass_at_1_mean"] * 100.0 for summary in summaries]
    functional_error = [summary["pass_at_1_standard_deviation"] * 100.0 for summary in summaries]
    produced: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(methods, security, yerr=security_error, capsize=4)
    axes[0].set_ylabel("security rate (%)")
    axes[0].set_title("Security rate by method")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(methods, functional, yerr=functional_error, capsize=4)
    axes[1].set_ylabel("pass@1 (%)")
    axes[1].set_title("Functional correctness by method")
    axes[1].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    target = _figure_path(config, name, "method_comparison")
    figure.savefig(target, dpi=200)
    plt.close(figure)
    produced.append(target)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.errorbar(security, functional, xerr=security_error, yerr=functional_error, fmt="o", capsize=3)
    for method, x_value, y_value in zip(methods, security, functional):
        axis.annotate(method, (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8)
    axis.set_xlabel("security rate (%)")
    axis.set_ylabel("pass@1 (%)")
    axis.set_title("Security and functional correctness")
    figure.tight_layout()
    target = _figure_path(config, name, "trade_off")
    figure.savefig(target, dpi=200)
    plt.close(figure)
    produced.append(target)

    per_category = payload.get("per_category")
    if per_category:
        categories = sorted(
            {cwe for values in per_category.values() for cwe in values},
            key=lambda item: int(item.split("-")[1]),
        )
        figure, axis = plt.subplots(figsize=(max(8, len(categories) * 0.8), 5))
        width = 0.8 / max(len(methods), 1)
        for position, method in enumerate(methods):
            values = [per_category.get(method, {}).get(cwe, {}).get("rate", 0.0) * 100.0 for cwe in categories]
            offsets = [index + position * width for index in range(len(categories))]
            axis.bar(offsets, values, width=width, label=method)
        axis.set_xticks([index + 0.4 for index in range(len(categories))])
        axis.set_xticklabels(categories, rotation=90)
        axis.set_ylabel("avoidance rate (%)")
        axis.set_title("Per-category avoidance rate")
        axis.legend()
        figure.tight_layout()
        target = _figure_path(config, name, "per_category")
        figure.savefig(target, dpi=200)
        plt.close(figure)
        produced.append(target)
    return produced


def _cross_model_figure(config: Config, name: str, payload: dict[str, Any]) -> list[Path]:
    models = payload.get("models", [])
    if not models:
        return []
    labels = [entry["label"] for entry in models]
    differences = [entry["security_rate_difference"] * 100.0 for entry in models]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, differences)
    axis.set_ylabel("security rate difference (percentage points)")
    axis.set_title("Cross-model improvement")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    target = _figure_path(config, name, "cross_model")
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return [target]


def _noise_figure(config: Config, name: str, payload: dict[str, Any]) -> list[Path]:
    conditions = payload.get("conditions", [])
    if not conditions:
        return []
    labels = [entry["condition"] for entry in conditions]
    values = [entry["security_rate"] * 100.0 for entry in conditions]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(len(labels)), values, marker="o")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45)
    axis.set_ylabel("security rate (%)")
    axis.set_title("Security rate under degraded reasoning")
    figure.tight_layout()
    target = _figure_path(config, name, "reasoning_noise")
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return [target]


def _sensitivity_figure(config: Config, name: str, payload: dict[str, Any]) -> list[Path]:
    sweeps = payload.get("sweeps", {})
    if not sweeps:
        return []
    figure, axes = plt.subplots(1, len(sweeps), figsize=(6 * len(sweeps), 5), squeeze=False)
    for position, (parameter, entries) in enumerate(sorted(sweeps.items())):
        axis = axes[0][position]
        values = [entry["value"] for entry in entries]
        security = [entry["security_rate"] * 100.0 for entry in entries]
        axis.plot(values, security, marker="o")
        axis.set_xlabel(parameter)
        axis.set_ylabel("security rate (%)")
        axis.set_title(f"Sensitivity to {parameter}")
    figure.tight_layout()
    target = _figure_path(config, name, "sensitivity")
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return [target]


def _cost_figure(config: Config, name: str, payload: dict[str, Any]) -> list[Path]:
    offline = payload.get("offline_cost", {}).get("stages_minutes", {})
    if not offline:
        return []
    stages = sorted(offline)
    minutes = [offline[stage] for stage in stages]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(stages, minutes)
    axis.set_ylabel("minutes")
    axis.set_title("Offline preparation cost")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    target = _figure_path(config, name, "offline_cost")
    figure.savefig(target, dpi=200)
    plt.close(figure)
    return [target]


_RENDERERS = {
    "method_comparison": _method_comparison_figures,
    "cross_model": _cross_model_figure,
    "reasoning_noise": _noise_figure,
    "hyperparameter_sensitivity": _sensitivity_figure,
    "cost_analysis": _cost_figure,
}


def render_figures(config: Config, names: Sequence[str] | None = None) -> list[Path]:
    declared = config.require_mapping("experiments")
    selected = list(names) if names else sorted(declared)
    produced: list[Path] = []
    for name in selected:
        path = _results_path(config, name)
        if not path.is_file():
            LOGGER.warning("no stored results for experiment '%s' at %s", name, path)
            continue
        kind = str(declared[name].get("kind", ""))
        renderer = _RENDERERS.get(kind)
        if renderer is None:
            continue
        produced.extend(renderer(config, name, read_json(path)))
    LOGGER.info("rendered %d figures", len(produced))
    return produced
