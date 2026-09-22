from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from reasonsec.config import Config
from reasonsec.intervention.policy import RecalibrationPolicy
from reasonsec.sae.model import SparseAutoencoder
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class OfflineCostReport:
    stages_minutes: dict[str, float] = field(default_factory=dict)

    @property
    def total_minutes(self) -> float:
        return float(sum(self.stages_minutes.values()))

    def as_dict(self) -> dict[str, object]:
        return {"stages_minutes": dict(self.stages_minutes), "total_minutes": self.total_minutes}


@dataclass
class InferenceCostReport:
    generation_milliseconds: float
    sae_milliseconds: float
    policy_milliseconds: float
    repetitions: int

    @property
    def overhead_milliseconds(self) -> float:
        return self.sae_milliseconds + self.policy_milliseconds

    @property
    def total_milliseconds(self) -> float:
        return self.generation_milliseconds + self.overhead_milliseconds

    @property
    def overhead_fraction(self) -> float:
        return self.overhead_milliseconds / self.total_milliseconds if self.total_milliseconds else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "generation_milliseconds": self.generation_milliseconds,
            "sae_milliseconds": self.sae_milliseconds,
            "policy_milliseconds": self.policy_milliseconds,
            "overhead_milliseconds": self.overhead_milliseconds,
            "total_milliseconds": self.total_milliseconds,
            "overhead_fraction": self.overhead_fraction,
            "repetitions": self.repetitions,
        }


def measure_inference_overhead(
    config: Config,
    autoencoder: SparseAutoencoder,
    policy: RecalibrationPolicy,
    activation: torch.Tensor,
    observation: torch.Tensor,
    generation_latencies_seconds: Sequence[float],
) -> InferenceCostReport:
    section = config.require_section("evaluation.cost")
    repetitions = section.require_int("repetitions")
    warmup = section.require_int("warmup_repetitions")
    device = torch.device(section.require_str("device"))

    autoencoder = autoencoder.to(device).eval()
    policy = policy.to(device).eval()
    activation = activation.to(device)
    observation = observation.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            autoencoder.decode(autoencoder.encode(activation))
            policy.act(observation, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repetitions):
            features = autoencoder.encode(activation)
            autoencoder.decode(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sae_milliseconds = (time.perf_counter() - started) * 1000.0 / repetitions

        started = time.perf_counter()
        for _ in range(repetitions):
            policy.act(observation, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        policy_milliseconds = (time.perf_counter() - started) * 1000.0 / repetitions

    generation_milliseconds = (
        float(sum(generation_latencies_seconds) / len(generation_latencies_seconds) * 1000.0)
        if generation_latencies_seconds
        else 0.0
    )
    report = InferenceCostReport(
        generation_milliseconds=generation_milliseconds,
        sae_milliseconds=sae_milliseconds,
        policy_milliseconds=policy_milliseconds,
        repetitions=repetitions,
    )
    LOGGER.info(
        "inference overhead %.3f ms of %.3f ms total (%.4f)",
        report.overhead_milliseconds,
        report.total_milliseconds,
        report.overhead_fraction,
    )
    return report
