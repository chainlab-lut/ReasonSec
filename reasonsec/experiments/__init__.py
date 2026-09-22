from __future__ import annotations

from typing import Any

from reasonsec.config import Config, ConfigError
from reasonsec.experiments.comparisons import run_avoidance_overlap, run_method_comparison
from reasonsec.experiments.cost_analysis import run_cost_analysis
from reasonsec.experiments.cross_model import run_cross_model
from reasonsec.experiments.robustness import (
    run_feature_label_agreement,
    run_hyperparameter_sensitivity,
    run_oracle_reliability,
    run_reasoning_noise,
)
from reasonsec.experiments.runner import ExperimentRunner, MethodEvaluation

_KINDS: dict[str, str] = {
    "method_comparison": "runner",
    "cost_analysis": "runner",
    "avoidance_overlap": "runner",
    "reasoning_noise": "runner",
    "oracle_reliability": "runner",
    "feature_label_agreement": "runner",
    "cross_model": "config",
    "hyperparameter_sensitivity": "config",
}


def available_experiments(config: Config) -> list[str]:
    return sorted(config.require_mapping("experiments").keys())


def run_experiment(config: Config, name: str, runner: ExperimentRunner | None = None) -> dict[str, Any]:
    section = config.require_section(f"experiments.{name}")
    kind = section.require_str("kind")
    if kind not in _KINDS:
        raise ConfigError(f"unknown experiment kind '{kind}'; available kinds are {sorted(_KINDS)}")
    if _KINDS[kind] == "config":
        if kind == "cross_model":
            return run_cross_model(config, name)
        return run_hyperparameter_sensitivity(config, name)
    active_runner = runner or ExperimentRunner(config)
    if kind == "method_comparison":
        return run_method_comparison(
            active_runner,
            name,
            [str(item) for item in section.require_list("methods")],
            section.require_str("reference_method"),
            section.require_bool("include_per_category"),
        )
    if kind == "cost_analysis":
        return run_cost_analysis(
            active_runner,
            name,
            [str(item) for item in section.require_list("methods")],
            section.require_str("reference_method"),
        )
    if kind == "avoidance_overlap":
        return run_avoidance_overlap(active_runner, name, [str(item) for item in section.require_list("methods")])
    if kind == "reasoning_noise":
        return run_reasoning_noise(active_runner, name)
    if kind == "oracle_reliability":
        return run_oracle_reliability(active_runner, name)
    return run_feature_label_agreement(active_runner, name)


__all__ = [
    "available_experiments",
    "run_experiment",
    "ExperimentRunner",
    "MethodEvaluation",
    "run_avoidance_overlap",
    "run_cost_analysis",
    "run_cross_model",
    "run_feature_label_agreement",
    "run_hyperparameter_sensitivity",
    "run_method_comparison",
    "run_oracle_reliability",
    "run_reasoning_noise",
]
