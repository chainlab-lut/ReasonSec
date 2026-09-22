from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reasonsec.config import Config
from reasonsec.data.preprocessing import PreprocessingReport
from reasonsec.types import CorpusSample
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CurationOutcome:
    retained: list[CorpusSample]
    rejected_specificity: list[CorpusSample]
    rejected_consistency: list[CorpusSample]

    def as_dict(self) -> dict[str, int]:
        return {
            "retained": len(self.retained),
            "rejected_cwe_specificity": len(self.rejected_specificity),
            "rejected_oracle_consistency": len(self.rejected_consistency),
        }


class CorpusCurator:
    def __init__(self, config: Config) -> None:
        section = config.require_section("reasoning.curation")
        self._require_specificity = section.require_bool("cwe_specificity_filter")
        self._require_consistency = section.require_bool("oracle_consistency_filter")
        self._require_non_empty_code = section.require_bool("require_non_empty_code")
        self._require_planning_segment = section.require_bool("require_planning_segment")
        self._safe_requires_mentioned_cwe = section.require_bool("safe_requires_mentioned_cwe")

    def passes_specificity(self, sample: CorpusSample) -> bool:
        if not self._require_specificity:
            return True
        return bool(sample.mentioned_cwes)

    def passes_consistency(self, sample: CorpusSample) -> bool:
        if not self._require_consistency:
            return True
        if sample.verdict.vulnerable:
            detected = sample.verdict.cwe
            if detected is None:
                return False
            return detected in sample.mentioned_cwes
        if self._safe_requires_mentioned_cwe:
            return bool(sample.mentioned_cwes)
        return True

    def curate(self, samples: Sequence[CorpusSample], report: PreprocessingReport | None = None) -> CurationOutcome:
        structural: list[CorpusSample] = []
        for sample in samples:
            if self._require_non_empty_code and not sample.chain.code.strip():
                continue
            if self._require_planning_segment and not sample.chain.planning_segment.strip():
                continue
            structural.append(sample)
        if report is not None:
            report.add("reasoning chain elicitation", len(samples) - len(structural), len(structural))

        specific = [sample for sample in structural if self.passes_specificity(sample)]
        specific_identities = {id(sample) for sample in specific}
        rejected_specificity = [sample for sample in structural if id(sample) not in specific_identities]
        if report is not None:
            report.add("cwe specificity filter", len(structural) - len(specific), len(specific))

        consistent = [sample for sample in specific if self.passes_consistency(sample)]
        consistent_identities = {id(sample) for sample in consistent}
        rejected_consistency = [sample for sample in specific if id(sample) not in consistent_identities]
        if report is not None:
            report.add("oracle consistency check", len(specific) - len(consistent), len(consistent))

        LOGGER.info(
            "curation retained %d of %d chains (%d failed specificity, %d failed consistency)",
            len(consistent),
            len(samples),
            len(rejected_specificity),
            len(rejected_consistency),
        )
        return CurationOutcome(
            retained=consistent,
            rejected_specificity=rejected_specificity,
            rejected_consistency=rejected_consistency,
        )
