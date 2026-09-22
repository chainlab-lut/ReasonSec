from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.experiments import available_experiments, run_experiment
from reasonsec.experiments.runner import ExperimentRunner
from reasonsec.pipeline import ReasonSecPipeline
from reasonsec.types import BenchmarkPrompt
from reasonsec.utils.io import read_json, write_json
from reasonsec.utils.logging import configure_logging, get_logger
from reasonsec.utils.seeding import seed_everything

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasonsec", description="ReasonSec pipeline and experiment driver")
    parser.add_argument("--config", required=True, help="path to the YAML configuration file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a configuration entry, for example --set rafs.alpha=0.5",
    )
    parser.add_argument("--log-level", default=None, help="logging level for this invocation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus = subparsers.add_parser("corpus", help="build the curated reasoning corpus")
    corpus.add_argument("--seed", type=int, default=None)

    vocabulary = subparsers.add_parser("vocabulary", help="extract the security concept vocabularies")
    vocabulary.add_argument("--seed", type=int, default=None)

    sae = subparsers.add_parser("sae", help="collect activations and train the sparse autoencoder")
    sae.add_argument("--seed", type=int, default=None)

    attribute = subparsers.add_parser("attribute", help="compute RAFS feature sets for a method variant")
    attribute.add_argument("--method", required=True)
    attribute.add_argument("--seed", type=int, default=None)

    policies = subparsers.add_parser("policies", help="train the recalibration policies for a method variant")
    policies.add_argument("--method", required=True)
    policies.add_argument("--seed", type=int, default=None)

    prepare = subparsers.add_parser("prepare", help="run every offline stage for the listed methods")
    prepare.add_argument("--methods", nargs="+", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one or more methods")
    evaluate.add_argument("--methods", nargs="+", required=True)
    evaluate.add_argument("--seeds", nargs="*", type=int, default=None)

    experiment = subparsers.add_parser("experiment", help="run a configured experiment")
    group = experiment.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="name of a configured experiment")
    group.add_argument("--all", action="store_true", help="run every configured experiment")

    generate = subparsers.add_parser("generate", help="generate code for prompts held in a JSON file")
    generate.add_argument("--method", required=True)
    generate.add_argument("--prompts", required=True, help="path to a JSON file holding a list of prompt records")
    generate.add_argument("--output", required=True)
    generate.add_argument("--seed", type=int, default=None)

    figures = subparsers.add_parser("figures", help="render figures from stored experiment results")
    figures.add_argument("--experiments", nargs="*", default=None)

    subparsers.add_parser("list-experiments", help="list the experiments declared in the configuration")

    return parser


def _load_config(arguments: argparse.Namespace) -> Config:
    config = Config.load(arguments.config, arguments.overrides)
    level = arguments.log_level or config.require_str("run.log_level")
    configure_logging(level, config.optional("run.log_file"))
    return config


def _pipeline(config: Config, seed: int | None) -> ReasonSecPipeline:
    resolved_seed = seed if seed is not None else config.require_int("run.seed")
    seed_everything(resolved_seed, config.require_bool("run.deterministic_algorithms"))
    overrides = {}
    if config.require_bool("run.seed_scoped_outputs"):
        overrides["run.output_directory"] = str(
            Path(str(config.require("run.output_directory"))).expanduser() / f"seed_{resolved_seed}"
        )
    return ReasonSecPipeline(config.derive(overrides), resolved_seed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = _load_config(arguments)
        command = arguments.command
        if command == "list-experiments":
            for name in available_experiments(config):
                print(name)
            return 0
        if command == "corpus":
            pipeline = _pipeline(config, arguments.seed)
            samples = pipeline.build_corpus()
            print(f"curated corpus size: {len(samples)}")
            return 0
        if command == "vocabulary":
            pipeline = _pipeline(config, arguments.seed)
            vocabulary = pipeline.build_vocabulary()
            print(f"vocabulary categories: {len(vocabulary)}")
            return 0
        if command == "sae":
            pipeline = _pipeline(config, arguments.seed)
            autoencoder = pipeline.train_autoencoder()
            print(f"sparse autoencoder features: {autoencoder.feature_dimension}")
            return 0
        if command == "attribute":
            pipeline = _pipeline(config, arguments.seed)
            from reasonsec.pipeline import MethodDefinition

            definition = MethodDefinition.from_config(config, arguments.method)
            feature_sets = pipeline.build_feature_sets(definition.attribution_key, pipeline.method_config(definition))
            for cwe, feature_set in sorted(feature_sets.items()):
                print(f"{cwe}: {len(feature_set.feature_indices)} features")
            return 0
        if command == "policies":
            pipeline = _pipeline(config, arguments.seed)
            from reasonsec.pipeline import MethodDefinition

            definition = MethodDefinition.from_config(config, arguments.method)
            policies = pipeline.train_policies(
                definition.policy_key, pipeline.method_config(definition), definition.attribution_key
            )
            print(f"trained policies: {len(policies)}")
            return 0
        if command == "prepare":
            runner = ExperimentRunner(config)
            runner.prepare(arguments.methods)
            return 0
        if command == "evaluate":
            runner = ExperimentRunner(config)
            runner.prepare(arguments.methods)
            for method in arguments.methods:
                evaluation = runner.evaluate_method(method, arguments.seeds)
                aggregate = evaluation.aggregate()
                write_json(runner.root / "results" / method / "aggregate.json", aggregate)
                print(
                    f"{method}: security rate {aggregate['security_rate_mean'] * 100:.2f}% "
                    f"pass@1 {aggregate['pass_at_1_mean'] * 100:.2f}%"
                )
            return 0
        if command == "experiment":
            names = available_experiments(config) if arguments.all else [arguments.name]
            runner = ExperimentRunner(config)
            for name in names:
                run_experiment(config, name, runner)
            return 0
        if command == "generate":
            pipeline = _pipeline(config, arguments.seed)
            records = read_json(arguments.prompts)
            prompts = [BenchmarkPrompt.from_dict(record) for record in records]
            generator = pipeline.build_generator(arguments.method)
            outputs = []
            for prompt in prompts:
                outcome = generator.generate(prompt)
                verdict = pipeline.oracle.evaluate(outcome.chain.code, prompt.language, prompt.cwe)
                outputs.append(
                    {
                        "identifier": prompt.identifier,
                        "applied_cwe": outcome.applied_cwe,
                        "planning_segment": outcome.chain.planning_segment,
                        "code": outcome.chain.code,
                        "verdict": verdict.as_dict(),
                        "latency_seconds": outcome.latency_seconds,
                    }
                )
            write_json(arguments.output, outputs)
            return 0
        if command == "figures":
            from reasonsec.figures import render_figures

            render_figures(config, arguments.experiments)
            return 0
        parser.error(f"unhandled command '{command}'")
        return 2
    except ConfigError as error:
        LOGGER.error("configuration error: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
