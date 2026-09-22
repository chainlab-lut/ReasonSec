from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from reasonsec.attribution.activations import FeatureMatrices, compute_feature_matrices
from reasonsec.attribution.rafs import ReasoningAnchoredFeatureScorer, group_rows_by_category
from reasonsec.attribution.validation import SemanticValidator, apply_validation, export_feature_labels
from reasonsec.baselines.prompting import build_prompting_generator
from reasonsec.baselines.sft_lora import (
    build_secure_examples,
    load_lora_adapter,
    train_lora_adapter,
    unload_lora_adapter,
)
from reasonsec.config import Config, ConfigError
from reasonsec.data.benchmarks import load_security_prompts
from reasonsec.data.cwe_catalog import CweCatalog
from reasonsec.data.preprocessing import PromptPreprocessor, stratified_split
from reasonsec.intervention.environment import RecalibrationEnvironment
from reasonsec.intervention.policy import RecalibrationPolicy
from reasonsec.intervention.ppo import PpoTrainer
from reasonsec.intervention.proxy import ProxyVulnerabilityClassifier, train_proxy_classifier
from reasonsec.intervention.reward import ReasoningCoherenceReward
from reasonsec.intervention.runtime import (
    PolicyStrategy,
    QuantileMappingStrategy,
    ReasonSecGenerator,
    ReplacementStrategy,
    SafeMeanStrategy,
    StateBuilder,
    ZeroAblationStrategy,
)
from reasonsec.models.embedder import SentenceEmbedder
from reasonsec.models.language_model import LanguageModel
from reasonsec.oracle import build_oracle
from reasonsec.oracle.base import SecurityOracle
from reasonsec.reasoning.curation import CorpusCurator
from reasonsec.reasoning.elicitation import ReasoningElicitor
from reasonsec.reasoning.parsing import ChainParser
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.reasoning.vocabulary import ConceptExtractor, SecurityConceptVocabulary
from reasonsec.sae.model import SparseAutoencoder
from reasonsec.sae.store import ActivationCollector, ActivationStore
from reasonsec.sae.train import train_sparse_autoencoder
from reasonsec.types import BenchmarkPrompt, CorpusSample, FeatureSet
from reasonsec.utils.io import ensure_directory, read_json, read_jsonl, write_json, write_jsonl
from reasonsec.utils.logging import get_logger
from reasonsec.utils.timing import StageTimer

LOGGER = get_logger(__name__)

_STRATEGIES = {
    "quantile_mapping": QuantileMappingStrategy,
    "zero": ZeroAblationStrategy,
    "safe_mean": SafeMeanStrategy,
}


@dataclass
class MethodDefinition:
    name: str
    kind: str
    strategy: str | None
    template: str
    overrides: dict[str, Any] = field(default_factory=dict)
    attribution_variant: str | None = None
    policy_variant: str | None = None

    @classmethod
    def from_config(cls, config: Config, name: str) -> "MethodDefinition":
        section = config.require_section(f"methods.{name}")
        return cls(
            name=name,
            kind=section.require_str("kind"),
            strategy=section.optional("strategy"),
            template=section.require_str("template"),
            overrides={str(key): value for key, value in (section.optional("overrides") or {}).items()},
            attribution_variant=section.optional("attribution_variant"),
            policy_variant=section.optional("policy_variant"),
        )

    @property
    def attribution_key(self) -> str:
        return self.attribution_variant or self.name

    @property
    def policy_key(self) -> str:
        return self.policy_variant or self.name


class ReasonSecPipeline:
    def __init__(self, config: Config, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.timings = StageTimer()
        self._root = ensure_directory(config.require_path("run.output_directory", must_exist=False))
        self._model: LanguageModel | None = None
        self._embedder: SentenceEmbedder | None = None
        self._oracle: SecurityOracle | None = None
        self._catalog: CweCatalog | None = None
        self._templates: TemplateLibrary | None = None
        self._parser: ChainParser | None = None
        self._corpus: list[CorpusSample] | None = None
        self._vocabulary: SecurityConceptVocabulary | None = None
        self._autoencoder: SparseAutoencoder | None = None
        self._store: ActivationStore | None = None
        self._matrices: FeatureMatrices | None = None
        self._feature_sets: dict[str, dict[str, FeatureSet]] = {}
        self._policies: dict[str, dict[str, RecalibrationPolicy]] = {}
        self._fine_tuned = False

    @property
    def root(self) -> Path:
        return self._root

    def attach(
        self,
        model: LanguageModel | None = None,
        embedder: SentenceEmbedder | None = None,
        oracle: SecurityOracle | None = None,
        catalog: CweCatalog | None = None,
        templates: TemplateLibrary | None = None,
    ) -> "ReasonSecPipeline":
        if model is not None:
            self._model = model
        if embedder is not None:
            self._embedder = embedder
        if oracle is not None:
            self._oracle = oracle
        if catalog is not None:
            self._catalog = catalog
        if templates is not None:
            self._templates = templates
        return self

    def release_fine_tuning(self) -> None:
        if self._fine_tuned:
            unload_lora_adapter(self.model)
            self._fine_tuned = False

    @property
    def model(self) -> LanguageModel:
        if self._model is None:
            self._model = LanguageModel(self.config)
        return self._model

    @property
    def embedder(self) -> SentenceEmbedder:
        if self._embedder is None:
            self._embedder = SentenceEmbedder(self.config)
        return self._embedder

    @property
    def oracle(self) -> SecurityOracle:
        if self._oracle is None:
            self._oracle = build_oracle(self.config)
        return self._oracle

    @property
    def catalog(self) -> CweCatalog:
        if self._catalog is None:
            self._catalog = CweCatalog.from_config(self.config)
        return self._catalog

    @property
    def templates(self) -> TemplateLibrary:
        if self._templates is None:
            self._templates = TemplateLibrary(self.config)
        return self._templates

    @property
    def parser(self) -> ChainParser:
        if self._parser is None:
            self._parser = ChainParser(self.config)
        return self._parser

    def path(self, *parts: str) -> Path:
        return self._root.joinpath(*parts)

    def security_prompts(self, dataset_key: str | None = None) -> list[BenchmarkPrompt]:
        key = dataset_key or self.config.require_str("pipeline.security_dataset")
        return load_security_prompts(self.config, key)

    def build_corpus(self) -> list[CorpusSample]:
        corpus_directory = ensure_directory(self.path("corpus"))
        curated_path = corpus_directory / "curated.jsonl"
        if curated_path.is_file() and not self.config.require_bool("pipeline.rebuild_corpus"):
            self._corpus = [CorpusSample.from_dict(record) for record in read_jsonl(curated_path)]
            LOGGER.info("loaded curated corpus with %d samples", len(self._corpus))
            return self._corpus

        with self.timings.stage("corpus"):
            prompts = self.security_prompts()
            preprocessor = PromptPreprocessor(self.config, self.model.count_tokens)
            filtered, report = preprocessor.run(prompts)
            elicitor = ReasoningElicitor(
                self.config, self.model, self.oracle, self.catalog, self.templates, self.parser
            )
            universe = sorted({prompt.cwe for prompt in filtered if prompt.cwe})
            samples = elicitor.elicit(filtered, universe=universe)
            write_jsonl(corpus_directory / "chains.jsonl", (sample.as_dict() for sample in samples))
            curator = CorpusCurator(self.config)
            outcome = curator.curate(samples, report)
            splits = stratified_split(
                outcome.retained,
                self.config.require_mapping("preprocessing.split.fractions"),
                self.config.require_bool("preprocessing.split.stratify_by_cwe"),
                self.config.require_int("preprocessing.split.seed"),
            )
            write_jsonl(curated_path, (sample.as_dict() for sample in outcome.retained))
            write_json(corpus_directory / "preprocessing_report.json", report.as_dict())
            write_json(
                corpus_directory / "split_sizes.json",
                {name: len(items) for name, items in splits.items()},
            )
        self._corpus = outcome.retained
        return self._corpus

    @property
    def corpus(self) -> list[CorpusSample]:
        if self._corpus is None:
            self.build_corpus()
        return self._corpus or []

    def split(self, name: str) -> list[CorpusSample]:
        return [sample for sample in self.corpus if sample.split == name]

    def attribution_samples(self) -> list[CorpusSample]:
        return self.split(self.config.require_str("pipeline.attribution_split"))

    def evaluation_prompts(self) -> list[BenchmarkPrompt]:
        return [sample.prompt for sample in self.split(self.config.require_str("pipeline.evaluation_split"))]

    def build_vocabulary(self) -> SecurityConceptVocabulary:
        vocabulary_path = self.path("vocabulary", "vocabulary.json")
        if vocabulary_path.is_file() and not self.config.require_bool("pipeline.rebuild_vocabulary"):
            self._vocabulary = SecurityConceptVocabulary.load(vocabulary_path)
            return self._vocabulary
        with self.timings.stage("vocabulary"):
            grouped: dict[str, list[CorpusSample]] = {}
            for sample in self.attribution_samples():
                if sample.label:
                    grouped.setdefault(sample.label, []).append(sample)
            extractor = ConceptExtractor(self.config, self.model, self.templates)
            vocabulary = extractor.build(grouped)
            vocabulary.save(vocabulary_path)
        self._vocabulary = vocabulary
        return vocabulary

    @property
    def vocabulary(self) -> SecurityConceptVocabulary:
        if self._vocabulary is None:
            self.build_vocabulary()
        return self._vocabulary or SecurityConceptVocabulary()

    def build_activation_store(self) -> ActivationStore:
        if self._store is not None:
            return self._store
        collector = ActivationCollector(self.config, self.model)
        self._store = collector.collect(self.attribution_samples())
        return self._store

    def train_autoencoder(self) -> SparseAutoencoder:
        autoencoder_path = self.path("sae", "sparse_autoencoder.pt")
        if autoencoder_path.is_file() and not self.config.require_bool("pipeline.retrain_sae"):
            self._autoencoder, _metadata = SparseAutoencoder.load(
                autoencoder_path, map_location=self.config.require_str("sae.device")
            )
            self._store = self.build_activation_store()
            return self._autoencoder
        store = self.build_activation_store()
        with self.timings.stage("sae"):
            autoencoder, _statistics = train_sparse_autoencoder(self.config, store, autoencoder_path, self.seed)
        self._autoencoder = autoencoder
        return autoencoder

    @property
    def autoencoder(self) -> SparseAutoencoder:
        if self._autoencoder is None:
            self.train_autoencoder()
        if self._autoencoder is None:
            raise ConfigError("the sparse autoencoder could not be prepared")
        return self._autoencoder

    def feature_matrices(self) -> FeatureMatrices:
        if self._matrices is not None:
            return self._matrices
        matrices_path = self.path("attribution", "feature_matrices.npz")
        if matrices_path.is_file() and not self.config.require_bool("pipeline.recompute_feature_matrices"):
            self._matrices = FeatureMatrices.load(matrices_path)
            return self._matrices
        store = self.build_activation_store()
        with self.timings.stage("feature_matrices"):
            matrices = compute_feature_matrices(self.config, store, self.autoencoder)
            matrices.save(matrices_path)
        self._matrices = matrices
        return matrices

    def build_feature_sets(self, variant: str, config: Config | None = None) -> dict[str, FeatureSet]:
        if variant in self._feature_sets:
            return self._feature_sets[variant]
        variant_config = config or self.config
        directory = ensure_directory(self.path("attribution", variant))
        feature_sets_path = directory / "feature_sets.json"
        if feature_sets_path.is_file() and not variant_config.require_bool("pipeline.recompute_feature_sets"):
            payload = read_json(feature_sets_path)
            feature_sets = {key: FeatureSet.from_dict(value) for key, value in payload.items()}
            self._feature_sets[variant] = feature_sets
            return feature_sets

        matrices = self.feature_matrices()
        scorer = ReasoningAnchoredFeatureScorer(variant_config, self.vocabulary)
        vulnerable_rows, safe_rows = group_rows_by_category(matrices, split=None)
        validator = SemanticValidator(
            variant_config, self.model, self.build_activation_store(), self.autoencoder, self.corpus, self.templates
        )
        feature_sets: dict[str, FeatureSet] = {}
        with self.timings.stage("rafs"):
            for cwe in sorted(vulnerable_rows):
                built = scorer.build_feature_set(cwe, matrices, vulnerable_rows[cwe], safe_rows)
                if built is None:
                    continue
                feature_set, _scores = built
                outcome = validator.validate(feature_set, self.vocabulary)
                feature_sets[cwe] = apply_validation(feature_set, outcome)
                write_json(directory / f"validation_{cwe}.json", outcome.as_dict())
        write_json(feature_sets_path, {cwe: item.as_dict() for cwe, item in feature_sets.items()})
        write_json(directory / "feature_labels.json", export_feature_labels(feature_sets))
        self._feature_sets[variant] = feature_sets
        return feature_sets

    def _training_prompts(self, cwe: str, config: Config) -> list[BenchmarkPrompt]:
        limit = config.require_int("intervention.training.prompts_per_category")
        samples = [sample for sample in self.attribution_samples() if sample.label == cwe]
        prompts = [sample.prompt for sample in samples]
        return prompts[:limit] if limit > 0 else prompts

    def _proxy_classifier(
        self, cwe: str, feature_set: FeatureSet, variant: str, config: Config
    ) -> ProxyVulnerabilityClassifier | None:
        if not config.require_bool("intervention.proxy_classifier.enabled"):
            return None
        directory = ensure_directory(self.path("policies", variant, "proxy"))
        path = directory / f"{cwe}.pt"
        if path.is_file() and not config.require_bool("pipeline.retrain_policies"):
            return ProxyVulnerabilityClassifier.load(path, map_location=config.require_str("intervention.proxy_classifier.device"))
        matrices = self.feature_matrices()
        vulnerable_rows = matrices.indices_for(cwe)
        safe_rows = matrices.indices_for(None)
        if not vulnerable_rows or not safe_rows:
            return None
        classifier, statistics = train_proxy_classifier(
            config,
            matrices.distributional[vulnerable_rows],
            matrices.distributional[safe_rows],
            feature_set.feature_indices,
            self.seed,
        )
        classifier.save(path, metadata=statistics.as_dict())
        return classifier

    def build_state_builder(self, config: Config, feature_sets: Mapping[str, FeatureSet]) -> StateBuilder:
        return StateBuilder(config, self.model, self.autoencoder, self.vocabulary, sorted(feature_sets))

    def train_policies(
        self,
        variant: str,
        config: Config | None = None,
        attribution_variant: str | None = None,
    ) -> dict[str, RecalibrationPolicy]:
        if variant in self._policies:
            return self._policies[variant]
        variant_config = config or self.config
        feature_sets = self.build_feature_sets(attribution_variant or variant, variant_config)
        directory = ensure_directory(self.path("policies", variant))
        state_builder = self.build_state_builder(variant_config, feature_sets)
        reward = ReasoningCoherenceReward(
            variant_config, self.embedder, self.model, self.vocabulary, feature_sets
        )
        policies: dict[str, RecalibrationPolicy] = {}
        statistics_payload: dict[str, Any] = {}
        for cwe, feature_set in feature_sets.items():
            if not feature_set.feature_indices:
                continue
            checkpoint = directory / f"{cwe}.pt"
            lower = torch.tensor(
                [feature_set.action_bounds[int(index)][0] for index in feature_set.feature_indices],
                dtype=torch.float32,
            )
            upper = torch.tensor(
                [feature_set.action_bounds[int(index)][1] for index in feature_set.feature_indices],
                dtype=torch.float32,
            )
            if checkpoint.is_file() and not variant_config.require_bool("pipeline.retrain_policies"):
                policy, _metadata = RecalibrationPolicy.load(
                    checkpoint, map_location=variant_config.require_str("intervention.ppo.device")
                )
                policies[cwe] = policy
                continue
            prompts = self._training_prompts(cwe, variant_config)
            if not prompts:
                LOGGER.warning("no training prompts available for %s; skipping policy training", cwe)
                continue
            policy = RecalibrationPolicy.from_config(
                variant_config, state_builder.state_dimension(len(feature_set.feature_indices)), lower, upper
            )
            generator = ReasonSecGenerator(
                config=variant_config,
                model=self.model,
                autoencoder=self.autoencoder,
                feature_sets={cwe: feature_set},
                state_builder=state_builder,
                strategy=None,
                parser=self.parser,
                templates=self.templates,
                elicitation_template=self._template_by_name(variant_config.require_str("intervention.training.template")),
            )
            environment = RecalibrationEnvironment(
                variant_config,
                generator,
                self.oracle,
                reward,
                feature_set,
                self._proxy_classifier(cwe, feature_set, variant, variant_config),
            )
            environment.prepare(prompts)
            trainer = PpoTrainer(variant_config, policy, environment)
            with self.timings.stage("ppo"):
                statistics = trainer.train(self.seed, checkpoint)
            statistics_payload[cwe] = statistics.as_dict()
            policies[cwe] = policy
        if statistics_payload:
            write_json(directory / "training_statistics.json", statistics_payload)
        self._policies[variant] = policies
        return policies

    def _template_by_name(self, name: str):
        library = self.templates
        available = {
            "elicitation": library.elicitation,
            "unstructured": library.unstructured,
            "zero_shot_security": library.zero_shot_security,
            "one_shot_security": library.one_shot_security,
            "chain_of_thought_security": library.chain_of_thought_security,
            "functional_completion": library.functional_completion,
        }
        if name not in available:
            raise ConfigError(f"unknown template '{name}'; available templates are {sorted(available)}")
        return available[name]

    def _build_strategy(
        self,
        definition: MethodDefinition,
        config: Config,
        feature_sets: Mapping[str, FeatureSet],
        state_builder: StateBuilder,
    ) -> ReplacementStrategy | None:
        if definition.strategy is None or definition.strategy == "none":
            return None
        if definition.strategy == "policy":
            policies = self.train_policies(definition.policy_key, config, definition.attribution_key)
            return PolicyStrategy(
                policies,
                state_builder,
                deterministic=config.require_bool("intervention.runtime.deterministic_action"),
                device=torch.device(config.require_str("intervention.ppo.device")),
            )
        if definition.strategy in _STRATEGIES:
            return _STRATEGIES[definition.strategy]()
        raise ConfigError(f"unknown replacement strategy '{definition.strategy}'")

    def method_config(self, definition: MethodDefinition) -> Config:
        return self.config.derive(definition.overrides)

    def build_generator(self, method: str):
        definition = MethodDefinition.from_config(self.config, method)
        config = self.method_config(definition)
        if definition.kind == "prompting":
            return build_prompting_generator(
                config,
                self.model,
                definition.template,
                exemplar_pool=self.attribution_samples(),
                templates=self.templates,
                parser=ChainParser(config),
            )
        if definition.kind == "fine_tuned":
            self.ensure_fine_tuned(config)
            return build_prompting_generator(
                config,
                self.model,
                definition.template,
                exemplar_pool=self.attribution_samples(),
                templates=self.templates,
                parser=ChainParser(config),
            )
        if definition.kind == "intervention":
            feature_sets = self.build_feature_sets(definition.attribution_key, config)
            state_builder = self.build_state_builder(config, feature_sets)
            strategy = self._build_strategy(definition, config, feature_sets, state_builder)
            return ReasonSecGenerator(
                config=config,
                model=self.model,
                autoencoder=self.autoencoder,
                feature_sets=feature_sets,
                state_builder=state_builder,
                strategy=strategy,
                parser=ChainParser(config),
                templates=self.templates,
                elicitation_template=self._template_by_name(definition.template),
            )
        raise ConfigError(f"unknown method kind '{definition.kind}' for method '{method}'")

    def ensure_fine_tuned(self, config: Config) -> None:
        if self._fine_tuned:
            return
        directory = self.path("baselines", "supervised_fine_tuning")
        if directory.joinpath("adapter_config.json").is_file() and not config.require_bool(
            "pipeline.retrain_fine_tuning"
        ):
            load_lora_adapter(self.model, directory)
        else:
            examples = build_secure_examples(config, self.attribution_samples(), self.templates, self.model)
            with self.timings.stage("supervised_fine_tuning"):
                train_lora_adapter(config, self.model, examples, directory, self.seed)
        self._fine_tuned = True

    def prepare_offline(self, methods: Sequence[str]) -> None:
        self.build_corpus()
        self.build_vocabulary()
        self.train_autoencoder()
        self.feature_matrices()
        for method in methods:
            definition = MethodDefinition.from_config(self.config, method)
            if definition.kind != "intervention":
                continue
            config = self.method_config(definition)
            self.build_feature_sets(definition.attribution_key, config)
            if definition.strategy == "policy":
                self.train_policies(definition.policy_key, config, definition.attribution_key)

    def offline_cost_minutes(self) -> dict[str, float]:
        return self.timings.as_minutes()
