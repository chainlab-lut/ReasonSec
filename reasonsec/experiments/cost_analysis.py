from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from reasonsec.evaluation.cost import OfflineCostReport, measure_inference_overhead
from reasonsec.evaluation.report import render_markdown_table, write_csv, write_report
from reasonsec.experiments.comparisons import run_avoidance_overlap
from reasonsec.experiments.runner import ExperimentRunner
from reasonsec.pipeline import MethodDefinition
from reasonsec.utils.io import write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def run_cost_analysis(runner: ExperimentRunner, name: str, methods: Sequence[str], reference_method: str) -> dict[str, Any]:
    config = runner.config
    runner.prepare(methods)
    evaluations = {method: runner.evaluate_method(method) for method in methods}
    seed = runner.seeds[0]
    pipeline = runner.pipeline(seed)
    offline = OfflineCostReport(stages_minutes=pipeline.offline_cost_minutes())

    definition = MethodDefinition.from_config(config, reference_method)
    method_config = pipeline.method_config(definition)
    feature_sets = pipeline.build_feature_sets(definition.attribution_key, method_config)
    policies = pipeline.train_policies(definition.policy_key, method_config, definition.attribution_key)
    latencies = [record.latency_seconds for record in evaluations[reference_method].security[seed].records]

    inference = None
    if policies and feature_sets:
        cwe = sorted(policies)[0]
        store = pipeline.build_activation_store()
        memmap = store.open_memmap()
        activation = torch.tensor(np.asarray(memmap[0:1], dtype=np.float32)).unsqueeze(0)
        observation = _observation_from_activation(
            pipeline, config, activation, feature_sets[cwe].feature_indices, len(feature_sets)
        )
        inference = measure_inference_overhead(
            config=config,
            autoencoder=pipeline.autoencoder,
            policy=policies[cwe],
            activation=activation,
            observation=observation,
            generation_latencies_seconds=latencies,
        )

    overlap = run_avoidance_overlap(runner, name, methods)
    payload: dict[str, Any] = {
        "name": name,
        "offline_cost": offline.as_dict(),
        "inference_cost": inference.as_dict() if inference is not None else {},
        "avoidance_overlap": overlap,
    }
    directory = runner.root / "experiments" / name
    offline_headers = ["Stage", "Minutes"]
    offline_rows = [[stage, f"{minutes:.1f}"] for stage, minutes in sorted(offline.stages_minutes.items())]
    offline_rows.append(["total", f"{offline.total_minutes:.1f}"])
    write_csv(directory / "offline_cost.csv", offline_headers, offline_rows)
    sections = [("Offline preparation cost", render_markdown_table(offline_headers, offline_rows))]
    if inference is not None:
        inference_headers = ["Component", "Milliseconds"]
        inference_rows = [
            ["generation", f"{inference.generation_milliseconds:.1f}"],
            ["sparse autoencoder pass", f"{inference.sae_milliseconds:.2f}"],
            ["policy network", f"{inference.policy_milliseconds:.2f}"],
            ["overhead fraction", f"{inference.overhead_fraction * 100.0:.2f}%"],
        ]
        write_csv(directory / "inference_cost.csv", inference_headers, inference_rows)
        sections.append(("Per-query cost", render_markdown_table(inference_headers, inference_rows)))
    if overlap:
        overlap_headers = ["Region", "Count"]
        overlap_rows = [[region, count] for region, count in sorted(overlap.get("regions", {}).items())]
        sections.append(("Avoidance overlap", render_markdown_table(overlap_headers, overlap_rows)))
    write_json(directory / "results.json", payload)
    write_report(directory / "report.md", f"Experiment: {name}", sections)
    return payload


def _observation_from_activation(pipeline, config, activation: "torch.Tensor", feature_indices, category_count: int):
    device = torch.device(config.require_str("evaluation.cost.device"))
    autoencoder = pipeline.autoencoder.to(device)
    with torch.no_grad():
        features = autoencoder.encode(activation.to(device)).reshape(-1)
    components = [features[list(feature_indices)].detach().cpu()]
    if config.require_bool("intervention.state.include_reasoning_embedding"):
        components.append(activation.reshape(-1).detach().cpu())
    if config.require_bool("intervention.state.include_suspicion_vector"):
        components.append(torch.zeros(category_count))
    return torch.cat(components).unsqueeze(0)
