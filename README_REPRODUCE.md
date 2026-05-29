# Reproduction Guide

This document describes how to reproduce the main GBSD experiments and the
paper-table artifacts used by the manuscript. The repository intentionally
keeps large checkpoints and full per-seed runtime folders out of GitHub. Those
artifacts should be regenerated locally, or archived separately as a release
artifact.

## 1. Environment

Create an environment from either file:

```powershell
conda env create -f environment.yml
conda activate gbsd
```

or:

```powershell
python -m pip install -r requirements.txt
```

Tested local development environment:

```text
Python: 3.13 in the local Code environment
PyTorch: 2.9.1+cu130
pandas: 2.3.3
GPU used for the archived summary tables: NVIDIA GeForce RTX 5060 Ti
```

The full experiment set is GPU-oriented. CPU execution is possible for smoke
tests, but full five-seed runs will be slow.

## 2. Repository Layout

```text
configs/          official protocol configuration files
experiments/      official stage runners and summarization entry points
runtime_engine/   migrated training engine and benchmark data
scripts/          reviewer-facing convenience wrappers
src/gbsd/         unified reporting, metrics, guard, and runtime adapters
paper_tools/      table generation, figure-source manifests, consistency checks
postprocess/      lightweight wrappers around summary and paper tools
results/          small official summary CSVs and generated paper tables
tests/            lightweight metric, guard, split, and consistency tests
legacy/           provenance map for historical 3.x development sources
```

Formal paper results should be read from `results/paper_tables/*.csv` and
`results/paper_tables/generated/*.csv`, not from transient runtime folders.

## 3. Smoke Test

Run this first to verify paths, imports, and the integrated runtime adapter:

```powershell
.\run_smoke_test.ps1 -PythonExecutable "python" -Seed 0 -McSamples 5
```

Expected outputs:

```text
results/smoke_test/
results/smoke_tables/
```

Smoke outputs are not paper results.

## 4. Official Main Experiments

The main protocol covers Laplace, Poisson, and Burgers inverse with seeds
`0,1,2,3,4`, a train / guard-validation / blind-test split, and guard decisions
made only on guard-validation data.

```powershell
python experiments/generate_all.py --stage main --execute --preset full --seeds 0,1,2,3,4 --n-mc 200 --python python
python experiments/summarize.py --input results/unified_blind_protocol
python paper_tools/generate_tables.py
python paper_tools/verify_consistency.py
```

Equivalent reviewer-facing wrapper:

```powershell
python scripts/run_all_main_experiments.py --preset full --seeds 0,1,2,3,4 --n-mc 200 --python python --summarize
```

Single-problem convenience runners are also provided:

```powershell
python scripts/run_laplace.py --preset full --seed 0 --n-mc 200 --python python
python scripts/run_poisson.py --preset full --seed 0 --n-mc 200 --python python
python scripts/run_burgers_inverse.py --preset full --seed 0 --n-mc 200 --python python
```

These single-problem scripts are useful for debugging and targeted reruns. The
paper tables should still be generated from the unified five-seed summaries.

## 5. Strong Baselines

The strong-baseline stage includes:

- Direct MC-Dropout PINN
- Deep Ensemble PINN
- structure-discovery baselines, including HAC, random grouping, magnitude
  grouping, and low-rank grouping where implemented
- GBSD rows injected from the main-blind summary during table generation

Run:

```powershell
python scripts/run_strong_baselines.py --preset full --seeds 0,1,2,3,4 --python python --summarize
```

or directly:

```powershell
python experiments/generate_all.py --stage baselines --execute --preset full --seeds 0,1,2,3,4 --python python
```

Expected official output root:

```text
results/unified_blind_protocol/strong_baselines/
```

Generated paper table:

```text
results/paper_tables/generated/table_5_4_strong_baselines.csv
```

## 6. Guard Ablation

The official ablation adapter runs the guard, UQ, reconstruction, and
zero-distillation ablation families together to keep the result contract
consistent. The guard table includes:

- always dense
- always structured
- accuracy-only guard
- compression-only guard
- full GBSD guard
- diagnostic random-source variant where available

Run:

```powershell
python scripts/run_guard_ablation.py --preset full --seeds 0,1,2,3,4 --n-mc 200 --python python --summarize
```

Generated paper table:

```text
results/paper_tables/generated/table_5_1_guard_ablation.csv
```

## 7. UQ Calibration Ablation

The UQ calibration ablation covers Poisson and Burgers inverse in the official
summary. Variants include:

- raw MC Dropout
- temperature scaling
- disagreement-only uncertainty proxy
- temperature plus disagreement
- full calibration

Run:

```powershell
python scripts/run_uq_ablation.py --preset full --seeds 0,1,2,3,4 --n-mc 200 --python python --summarize
```

Generated paper table:

```text
results/paper_tables/generated/table_5_3_uq_ablation.csv
```

## 8. Structured Reconstruction Ablation

The reconstruction ablation is a selected-component diagnostic, not an
exhaustive ablation over every possible loss term. It includes:

- full structured reconstruction
- without PDE residual loss
- without boundary or initial loss
- without distillation / dense-student anchor
- without hard boundary condition where applicable

Run:

```powershell
python scripts/run_reconstruction_ablation.py --preset full --seeds 0,1,2,3,4 --n-mc 200 --python python --summarize
```

Generated appendix table:

```text
results/paper_tables/generated/appendix_reconstruction_ablation.csv
```

## 9. Structure Diagnostics

Structure-discovery diagnostics are generated by the baseline stage. They cover
the implemented HAC, random, magnitude, and low-rank grouping baselines under
the unified blind-test metric definitions.

```powershell
python scripts/run_structure_diagnostics.py --preset full --seeds 0,1,2,3,4 --python python --summarize
```

This wrapper runs only the structure-discovery baseline adapter. If the strong
baseline table is needed, run `scripts/run_strong_baselines.py` as well so that
the UQ baselines are present.

## 10. Threshold Sensitivity and Structure Stability

The sensitivity stage covers source-selection thresholds, minimum compression,
clustering threshold sweeps, and outcome-level structure stability.

```powershell
python experiments/generate_all.py --stage sensitivity --execute --preset full --seeds 0,1,2,3,4 --n-mc 200 --python python
python experiments/summarize.py --input results/unified_blind_protocol
python paper_tools/generate_tables.py
```

If interrupted, resume missing sensitivity outputs:

```powershell
.\resume_missing_sensitivity.ps1 -PythonExecutable "python"
```

The current repository does not claim a full sensitivity analysis over every
UQ temperature or disagreement-weight hyperparameter.

## 11. Runtime and Parameter Analysis

Runtime and parameter summaries are derived from official main runs:

```powershell
python experiments/generate_all.py --stage runtime --execute --preset full --seeds 0,1,2,3,4 --python python
python experiments/summarize.py --input results/unified_blind_protocol
python paper_tools/generate_tables.py
```

Generated paper table:

```text
results/paper_tables/generated/table_4_4_runtime_params.csv
```

## 12. Generate Paper Tables

After official per-seed outputs are available:

```powershell
python experiments/summarize.py --input results/unified_blind_protocol
python paper_tools/generate_tables.py
```

Main generated files:

```text
results/paper_tables/generated/table_4_2_main_results.csv
results/paper_tables/generated/table_4_4_runtime_params.csv
results/paper_tables/generated/table_5_1_guard_ablation.csv
results/paper_tables/generated/table_5_3_uq_ablation.csv
results/paper_tables/generated/table_5_4_strong_baselines.csv
```

## 13. Generate Figure Source Manifest

The current GitHub package does not regenerate final manuscript PNG/PDF/SVG
panels. Instead, it writes a source manifest that maps each manuscript figure to
the official summary CSV used to create it. This prevents accidental reuse of
legacy local figures.

```powershell
python paper_tools/generate_figures.py
```

Expected output:

```text
results/paper_figures/figure_source_manifest.json
```

Use the listed summary files as the source data for final plotting or external
layout software.

## 14. Consistency and Submission Checks

Run before using any generated table in a manuscript:

```powershell
python paper_tools/verify_consistency.py
python paper_tools/verify_submission_assets.py
python paper_tools/export_submission_assets.py
```

Expected output:

```text
results/submission_assets/
```

The export folder is ignored by Git because it is a generated artifact.

## 15. Tests

If `pytest` is installed:

```powershell
python -m pytest tests -q
```

If `pytest` is not installed, the test files are small enough to inspect or run
manually from Python. The repository tests cover metrics, guard split usage,
summary-file mapping, and basic result schema checks.

## 16. Expected Runtime

Full five-seed runs are expensive. Approximate runtimes depend on GPU and
driver versions. Use `--preset smoke` or `--preset quick_check` for path tests.
Only `--preset full` should be used for paper-quality runs.

## 17. Data and Artifact Policy

Small summary CSV files and source-data manifests are included. Full per-seed
folders, raw runtime logs, model checkpoints, and large binary outputs should be
regenerated or released as external artifacts, for example through Zenodo.

Do not use legacy `3.x` paths as official paper-result sources. They are kept
only in `legacy/source_map.csv` for provenance.

