# ReasonSec

Reference implementation of **ReasonSec: Reasoning-Guided Latent Feature Attribution and Adaptive Activation Intervention for Secure Code Generation**.

ReasonSec turns a code model's own security reasoning into an operational control signal. It elicits a structured five-phase reasoning chain with a dedicated *Security Planning* phase, scores sparse-autoencoder features with **Reasoning-Anchored Feature Scoring (RAFS)**, and recalibrates the selected features during decoding with a reinforcement-learned policy trained under the **Reasoning-Coherence Reward (RCR)**.

The repository contains the complete framework: corpus construction, security concept vocabularies, sparse-autoencoder training, RAFS attribution with semantic validation, PPO policy training, the inference-time activation interceptor, every baseline and ablation reported in the paper, the evaluation harness (security rate, pass@1, multiple-choice accuracy, statistical tests, cost measurements), the robustness studies, and figure rendering.

No datasets, model weights, rule sets or results are bundled. Every path is supplied through the configuration file, and every hyperparameter is read from it — the code contains no default values of its own and no example data.

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[quantization]"    # optional, required only for 4-bit loading
```

Python 3.10 or newer is required. `semgrep` (and, for the alternative oracle, `codeql`) must be available as executables.

## 2. What you have to provide

ReasonSec runs only on real data. Before the first run, obtain the following and record their locations in the configuration:

| Resource | Purpose | Configuration key |
| --- | --- | --- |
| Open-weight causal language model | the generator that is inspected and intervened upon | `model.name_or_path` |
| Sentence encoder | dense embeddings for the reasoning–code coherence reward | `embedder.name_or_path` |
| MITRE CWE catalogue (`cwec_*.xml`) | CWE names and alternate terms used by the specificity filter | `cwe_catalog.path` |
| Primary security benchmark | prompts for corpus construction and security evaluation | `datasets.cyberseceval2.path` |
| Secondary security benchmark | cross-benchmark security evaluation | `datasets.securityeval.path` |
| Instruction-following code benchmark | pass@1 utility evaluation | `datasets.instructhumaneval.path` |
| Multiple-choice knowledge benchmark | second utility proxy | `datasets.mmlu.splits.*` |
| Semgrep rule sources | static half of the security oracle | `oracle.semgrep.rule_sources` |
| Regular-expression rule files | pattern half of the security oracle | `oracle.regex.rule_paths` |
| CodeQL query suite (optional) | alternative oracle for the oracle-reliability study | `oracle.codeql.query_suite` |

Every path in `configs/default.yaml` is written as `${env:VARIABLE:/absolute/path/...}`, so you can either edit the file or export the environment variables:

```bash
export REASONSEC_MODEL=/models/llama-3.1-8b-instruct
export REASONSEC_EMBEDDER=/models/all-mpnet-base-v2
export REASONSEC_CWE_CATALOG=/data/cwe/cwec_latest.xml
export REASONSEC_CYBERSECEVAL2=/data/cyberseceval2/instruct.json
export REASONSEC_SECURITYEVAL=/data/securityeval/dataset.jsonl
export REASONSEC_INSTRUCTHUMANEVAL=/data/instructhumaneval/test.jsonl
export REASONSEC_MMLU_TEST=/data/mmlu/test
export REASONSEC_MMLU_DEV=/data/mmlu/dev
export REASONSEC_SEMGREP_RULES=/data/rules/semgrep
export REASONSEC_REGEX_RULES=/data/rules/insecure_code_detector
export REASONSEC_OUTPUT_DIRECTORY=/runs/reasonsec
export REASONSEC_DEVICE=cuda
```

Datasets are read through explicit field maps, so any release layout can be used without touching the code. For example, `datasets.cyberseceval2.fields` maps the record keys holding the prompt, the language and the CWE label; `datasets.securityeval.cwe_from_identifier_pattern` recovers the CWE from the sample identifier when the records carry no label field.

The regular-expression oracle reads JSON, JSON Lines or YAML rule files. `oracle.regex.fields` names the keys that hold the pattern, the CWE, the language, the rule identifier and the message, so rule sets published in different shapes are loaded without conversion.

## 3. Running the framework

All commands take `--config` and accept `--set key.path=value` overrides.

```bash
reasonsec --config configs/default.yaml corpus            # stages 1-8: preprocessing, elicitation, curation, splits
reasonsec --config configs/default.yaml vocabulary        # security concept vocabularies V_k
reasonsec --config configs/default.yaml sae               # activation collection and sparse-autoencoder training
reasonsec --config configs/default.yaml attribute --method reasonsec   # RAFS feature sets F_k
reasonsec --config configs/default.yaml policies --method reasonsec    # PPO recalibration policies
reasonsec --config configs/default.yaml evaluate --methods reasonsec thea unmodified
reasonsec --config configs/default.yaml experiment --name rq1_sae_baselines
reasonsec --config configs/default.yaml experiment --all
reasonsec --config configs/default.yaml figures
```

`prepare` runs every offline stage that the listed methods need:

```bash
reasonsec --config configs/default.yaml prepare --methods reasonsec thea oracle_only
```

`generate` applies a trained method to prompts held in a JSON file (a list of records with `identifier`, `prompt`, `language`, `benchmark` and optionally `cwe`):

```bash
reasonsec --config configs/default.yaml generate --method reasonsec --prompts prompts.json --output generations.json
```

Artifacts are written under `run.output_directory` (per seed when `run.seed_scoped_outputs` is true):

```
corpus/          curated.jsonl, chains.jsonl, preprocessing_report.json, split_sizes.json
vocabulary/      vocabulary.json
sae/             sparse_autoencoder.pt, sparse_autoencoder.statistics.json
activations/     activations.npy, index.json
attribution/     feature_matrices.npz, <variant>/feature_sets.json, feature_labels.json, validation_<cwe>.json
policies/        <variant>/<cwe>.pt, proxy/<cwe>.pt, training_statistics.json
results/         <method>/seed_<n>/{security,secondary_security,functional,knowledge}.json
experiments/     <experiment>/{results.json,report.md,*.csv}
figures/         rendered PNG figures
```

## 4. How the framework is organised

| Stage | Module | What it does |
| --- | --- | --- |
| Preprocessing | `reasonsec/data/preprocessing.py` | eight-stage pipeline: ingestion, language filter, CWE-label filter, length filter, cyclomatic-complexity filter, code normalisation, then elicitation and the two curation filters; stratified corpus split |
| Reasoning chains | `reasonsec/reasoning/` | five-phase SDLC elicitation, phase segmentation with character spans, CWE-specificity and oracle-consistency filters, concept vocabulary extraction |
| Security oracle | `reasonsec/oracle/` | Semgrep and regular-expression components combined into the insecure-code detector, optional CodeQL oracle, on-disk verdict cache |
| Sparse autoencoder | `reasonsec/sae/` | residual-stream activation store with planning and code spans, ReLU autoencoder with L1 sparsity, L0 / explained-variance / dead-feature statistics |
| Attribution | `reasonsec/attribution/` | per-sample feature aggregates, histogram overlap distance, reasoning-conditioned alignment, RAFS interpolation, adaptive candidate count, semantic validation from top activating contexts |
| Intervention | `reasonsec/intervention/` | composite state, bounded squashed-Gaussian policy, RCR reward, proxy classifier, single-step environment, PPO trainer, decoding-time recalibration hook |
| Baselines | `reasonsec/baselines/` | prompting variants and LoRA supervised fine-tuning on the curated safe subset |
| Evaluation | `reasonsec/evaluation/` | security rate, sandboxed pass@1, multiple-choice accuracy, McNemar and paired-t tests, Cohen's kappa, offline and per-query cost |
| Experiments | `reasonsec/experiments/` | the five research questions, the ablation grid and the robustness studies |

### The activation edit

At layer `model.layer` the residual stream `h` is encoded to sparse features `z`, the selected features are replaced by the policy's action, the result is decoded, and the reconstruction error of the original activation is added back:

```
z  = E(h)
z' = z with z'_j = delta_j for every selected feature j
h' = D(z') + (h - D(E(h)))
```

`intervention.runtime.preserve_reconstruction_error` controls the final term and `intervention.runtime.apply_positions` chooses whether the edit is applied at every position of the forward pass or only at the newest one.

### Generation flow at inference time

1. The five-phase template is rendered and the model generates the reasoning chain, stopping once the Implementation heading appears (`intervention.runtime.trace_stop_markers`).
2. The Security Planning span is located, the residual stream is captured over the full context, and the state `(z, e_s, c)` is assembled: sparse features at the decision position, mean-pooled planning-span representation, and the binary CWE suspicion vector.
3. The suspected categories are ranked by the number of matched vocabulary phrases, ties are broken by the lower CWE identifier, and the winning category's policy produces one bounded replacement value per selected feature.
4. Generation of the implementation continues from the same sequence with the recalibration hook active.

### Method registry

Methods are declared in the `methods` section of the configuration. Each entry names a `kind` (`prompting`, `fine_tuned` or `intervention`), a replacement `strategy` (`policy`, `quantile_mapping`, `zero`, `safe_mean` or none), a prompt `template`, and optional `overrides` that are applied to a derived copy of the configuration before the offline artifacts for that method are built.

| Method | Description |
| --- | --- |
| `unmodified` | the base model on the raw benchmark prompt |
| `zero_shot_security`, `one_shot_security`, `cot_security` | prompting baselines; the one-shot exemplar is drawn from the curated safe corpus, never from a fixed string |
| `sdlc_prompting` | the five-phase template with no activation intervention |
| `sft_lora` | LoRA supervised fine-tuning on the safe subset of the attribution split |
| `thea`, `thea_zero` | histogram-only attribution with a fixed feature count, combined with quantile-mapping or zero-ablation replacement |
| `oracle_only` | RAFS attribution with a policy trained on the oracle term alone |
| `reasonsec` | the full framework |
| `n_sdlc`, `n_rafs`, `n_adaptk`, `n_rcc`, `n_fsf`, `n_semval` | ablations removing the structured template, the alignment term, the adaptive feature count, the coherence reward, the fidelity reward and semantic validation |
| `rafs_only`, `rcr_only` | isolating variants: RAFS attribution with an oracle-only reward, and the full reward with random feature selection |

`attribution_variant` and `policy_variant` let several methods share offline artifacts: the ablations that change only the reward reuse the RAFS feature sets of `reasonsec` while training their own policies.

### Experiments

| Experiment | Kind | Contents |
| --- | --- | --- |
| `rq1_sae_baselines` | `method_comparison` | comparison with the sparse-autoencoder baselines, per-CWE avoidance table, McNemar tests |
| `rq2_prompting_and_fine_tuning` | `method_comparison` | prompting and fine-tuning comparison on both security benchmarks, with pass@1 and knowledge accuracy |
| `rq3_ablation` | `method_comparison` | the full ablation grid |
| `rq4_cross_model` | `cross_model` | one complete pipeline per model entry, reporting the difference against each model's own baseline |
| `rq5_cost` | `cost_analysis` | measured offline stage times, per-query autoencoder and policy overhead, three-way avoidance overlap |
| `robustness_reasoning` | `reasoning_noise` | Security Planning corruption sweep plus the forced-incorrect-category condition |
| `robustness_oracle` | `oracle_reliability` | re-scoring every stored generation with the alternative oracle, agreement and per-tool rates |
| `robustness_sensitivity` | `hyperparameter_sensitivity` | sweeps over the reward weights and the RAFS interpolation weight, each retrained from scratch |
| `robustness_feature_labels` | `feature_label_agreement` | exports the feature labels for human rating and computes agreement and Cohen's kappa once ratings are supplied |

## 5. Configuration reference

The configuration is strict: every key the code reads must be present, and a missing or null entry raises `ConfigError` naming the key. Files may include one another through a top-level `include:` list, and `${env:NAME:fallback}` substitution is applied throughout. Relative paths are resolved against the directory holding the configuration file, so the framework can be run from any working directory.

| Section | Key settings |
| --- | --- |
| `run` | seed, seed list, output directory, logging, whether artifacts are scoped per seed |
| `cwe_catalog` | catalogue path, phrase-matching mode and thresholds, optional supplementary term file |
| `datasets` | one block per benchmark: paths, format, field map, language defaults and metadata fields |
| `model` | model path, layer, hook module path, quantisation, chat template, generation settings, token budgets |
| `embedder` | sentence encoder path, pooling, normalisation |
| `oracle` | primary oracle definition, component list, rule sources, CWE-selection strategy, verdict cache, language extensions |
| `preprocessing` | language allowlist, length and complexity thresholds, decision-token patterns, comment patterns, split fractions |
| `reasoning` | phase headings and patterns, template paths, curation filters, concept-extraction limits |
| `sae` | expansion factor, sparsity coefficient, optimisation settings, activation-store spans and dtype |
| `rafs` | `alpha`, histogram rule, aggregation spans, adaptive selection bounds, safe-distribution quantiles, semantic validation |
| `intervention` | state composition, policy shape, reward weights, proxy classifier, PPO settings, runtime behaviour |
| `baselines` | prompting budget, LoRA and supervised fine-tuning settings |
| `evaluation` | benchmark selection, sandboxed execution, knowledge-evaluation prompt format, statistical tests, cost measurement |
| `methods`, `experiments` | the registries described above |

The shipped values follow the paper: layer 19, expansion factor 8, sparsity coefficient 1e-3, 5 autoencoder epochs at batch size 256 and learning rate 1e-4, `alpha = 0.25`, reward weights `beta_oracle = 1.0`, `beta_coherence = 0.6`, `beta_fidelity = 0.4`, `beta_deviation = 0.1`, proxy blend `0.25`, a 128-token code prefix for the coherence term, 5 PPO epochs at learning rate 3e-4, a 350-token generation cap, a 70/30 attribution/evaluation split and seeds 42, 123 and 456.

## 6. Executing generated code

pass@1 requires running model-generated programs. This is disabled by default. Enable it only inside a sandbox:

```yaml
evaluation:
  utility:
    execution:
      enabled: true
      sandbox_command: ["firejail", "--quiet", "--net=none", "--private"]
```

`sandbox_command` is prepended to the interpreter invocation, `timeout_seconds` bounds each program, and `environment` replaces the environment passed to the child process. If the functional evaluation is enabled while execution is not, the runner refuses to start rather than failing part-way through.

## 7. Reproducing a full run

```bash
reasonsec --config configs/default.yaml prepare --methods unmodified thea thea_zero oracle_only reasonsec
reasonsec --config configs/default.yaml experiment --name rq1_sae_baselines
reasonsec --config configs/default.yaml experiment --name rq2_prompting_and_fine_tuning
reasonsec --config configs/default.yaml experiment --name rq3_ablation
reasonsec --config configs/default.yaml experiment --name rq5_cost
reasonsec --config configs/default.yaml experiment --name robustness_reasoning
reasonsec --config configs/default.yaml figures
```

Cross-model generalisation runs one complete pipeline per entry of `experiments.rq4_cross_model.models`; add one entry per architecture, each with its own `model.name_or_path` and `model.layer`.

Every numeric result is computed from the configured datasets and the configured oracle at run time. The repository stores no measurements from the paper, and nothing in the code substitutes for a missing dataset, oracle or model.

## 8. Citation

```bibtex
@article{sardar2026reasonsec,
  title   = {ReasonSec: Reasoning-Guided Latent Feature Attribution and Adaptive Activation Intervention for Secure Code Generation},
  author  = {Sardar, Bilal and Kumar, Prabhat and Islam, Shareeful and Islam, Najmul and Papastergiou, Spyridon},
  journal = {ACM Transactions on Software Engineering and Methodology},
  year    = {2026}
}
```
