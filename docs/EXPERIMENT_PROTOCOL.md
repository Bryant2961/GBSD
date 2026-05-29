# Experiment Protocol

This repository uses a unified blind-test protocol for the official GBSD paper
tables. The historical development folders are not paper-result labels.

## Benchmarks

The official benchmark set contains:

1. Laplace
2. Poisson
3. Burgers inverse

## Official Seeds

All official main experiments, strong baselines, and key ablations use:

```text
0, 1, 2, 3, 4
```

## Splits

Each problem and seed uses a train / guard-validation / blind-test split.

- Training data are used for teacher PINN training, dense Bayesian student
  distillation, structure discovery, and structured reconstruction.
- Guard-validation data are used for source-selection decisions.
- Blind-test data are used for final reported metrics.

The guard must not read blind-test labels. The consistency tests include a
guard split check.

## Main GBSD Pipeline

The official main pipeline is:

```text
Teacher PINN
-> Dense Bayesian student with MC Dropout
-> Structure discovery
-> Structured reconstruction
-> Guarded source selection
-> Blind-test evaluation
```

The final source is selected by the accuracy-compression guard. Structured
models are treated as candidates, not default final predictors.

## Metrics

Official summary files use shared metric definitions for:

- dense rL2
- structured rL2
- final rL2
- compression ratio
- Coverage95
- Gaussian NLL
- average interval width
- error-standard-deviation correlation
- Burgers inverse parameter error when applicable

Fields that are not physically meaningful for a variant are left blank in CSV
outputs and should not be interpreted as zero.

## Baselines and Diagnostics

The official strong-baseline stage includes direct MC-Dropout PINN, Deep
Ensemble PINN, and structure-discovery baselines. Structure diagnostics are
reported under the same blind-test metric definitions.

The reconstruction ablation is a selected-component diagnostic. It should not
be described as an exhaustive ablation of every possible reconstruction term.

Threshold sensitivity covers source-selection and clustering thresholds. The
current package does not claim a complete UQ hyperparameter sensitivity study
over all temperature and disagreement-weight choices.

## Official Result Contract

Formal per-seed outputs are written to:

```text
results/unified_blind_protocol/{experiment}/{problem}/{variant}/seed_{seed}/
```

Each official per-seed folder should contain a resolved configuration, split
manifest, predictions for guard and blind subsets, guard decision metadata,
metrics, timing information, model-size metadata, and a run manifest when that
information is available from the migrated runtime engine.

Paper tables must be generated from the summary CSV files in
`results/paper_tables/`, not from transient runtime folders.

