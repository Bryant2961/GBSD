# Guarded Bayesian Structured Distillation for PDE Learning

This repository contains the reproducibility code for **Guarded Bayesian
Structured Distillation (GBSD)**, a Bayesian structured-distillation framework
for PDE learning. GBSD trains a dense Bayesian student, reconstructs a compact
structured candidate, and accepts the structured candidate only when it passes
an accuracy-compression guard on a held-out guard-validation split.

The repository is organized for paper reproduction rather than historical
development. Large checkpoints and full per-seed result folders are intentionally
excluded from the GitHub package.

## What Is Included

- `configs/`: official problem, experiment, baseline, and ablation protocols.
- `src/gbsd/`: unified reporting, evaluation, guard, and runtime-adapter code.
- `runtime_engine/`: integrated training engine, benchmark data, model modules,
  and baseline runners needed to reproduce experiments.
- `experiments/`: top-level experiment entry points.
- `scripts/`: reviewer-facing wrappers for main experiments, baselines,
  ablations, and diagnostics.
- `paper_tools/`: scripts for paper tables, figure manifests, consistency
  checks, and submission-asset export.
- `postprocess/`: compatibility wrappers for summary, table, and figure-source
  generation.
- `results/paper_tables/`: small CSV summaries and generated paper-ready tables.
- `results/paper_figures/`: figure source manifest.
- `docs/`: experiment protocol, data availability, Zenodo notes, and repository
  checklist.
- `tests/`: lightweight consistency and metric tests.
- `legacy/`: source map for historical experiments; legacy outputs are not used
  as official paper labels.

## What Is Not Included

The following are intentionally excluded and should not be committed:

- `results/unified_blind_protocol/` full official per-seed outputs
- `runtime_engine/Results/` transient runtime outputs
- model checkpoints (`*.pth`, `*.pt`, `*.ckpt`)
- smoke-test outputs, render caches, Word/LibreOffice temporary files
- historical `3.x` development folders

## Benchmarks

The official benchmark set contains:

- Laplace
- Poisson
- Burgers inverse

The official paper protocol uses five random seeds (`0,1,2,3,4`) and a
train / guard-validation / blind-test split. Guard decisions are made only from
guard-validation data, while final metrics are reported on blind-test data.

## Installation

Create a Python environment with PyTorch, NumPy, SciPy, pandas, matplotlib, and
scikit-learn. A minimal package list is provided:

```powershell
python -m pip install -r requirements.txt
```

For GPU runs, install a PyTorch build compatible with your CUDA driver before
running the full training protocol.

## Quick Smoke Test

Run this before any long experiment:

```powershell
cd <repo>
.\run_smoke_test.ps1 -PythonExecutable "<PYTHON>" -Seed 0 -McSamples 5
```

The smoke test writes only to `results/smoke_test/` and `results/smoke_tables/`.
It does not alter formal paper-result summaries.

## Main Experiment

For the main GBSD experiment only:

```powershell
cd <repo>
<PYTHON> experiments/generate_all.py --stage main --execute --preset full --seeds 0,1,2,3,4 --n-mc 200 --python <PYTHON>
<PYTHON> experiments/summarize.py --input results/unified_blind_protocol
<PYTHON> paper_tools/generate_tables.py
<PYTHON> paper_tools/verify_consistency.py
```

This is the minimum reproducibility path for the core claims in the paper.

The same main protocol can be launched through a reviewer-facing wrapper:

```powershell
<PYTHON> scripts/run_all_main_experiments.py --preset full --seeds 0,1,2,3,4 --n-mc 200 --python <PYTHON> --summarize
```

For detailed reproduction instructions, see `README_REPRODUCE.md`.

## Optional Supplementary Experiments

The repository also keeps optional runners for the supplement and reviewer
checks. They are not required for a quick reproduction of the main experiment,
but they support the JCP-style evidence chain:

```powershell
<PYTHON> experiments/run_runtime_params.py --execute --preset full --seeds 0,1,2,3,4 --python <PYTHON>
<PYTHON> experiments/generate_all.py --stage baselines --execute --preset full --seeds 0,1,2,3,4 --python <PYTHON>
<PYTHON> experiments/run_ablation.py --execute --preset full --seeds 0,1,2,3,4 --n-mc 200 --python <PYTHON>
<PYTHON> experiments/run_sensitivity.py --execute --preset full --seeds 0,1,2,3,4 --n-mc 200 --python <PYTHON>
<PYTHON> experiments/summarize.py --input results/unified_blind_protocol
<PYTHON> paper_tools/generate_tables.py
<PYTHON> paper_tools/generate_figures.py
<PYTHON> paper_tools/verify_consistency.py
<PYTHON> paper_tools/export_submission_assets.py
```

If a sensitivity run is interrupted, resume only missing official outputs:

```powershell
.\resume_missing_sensitivity.ps1 -PythonExecutable "<PYTHON>"
```

## Paper Tables

Generated paper-ready tables are stored in:

```text
results/paper_tables/generated/
```

Important files include:

- `table_4_2_main_results.csv`
- `table_4_4_runtime_params.csv`
- `table_5_1_guard_ablation.csv`
- `table_5_3_uq_ablation.csv`
- `table_5_4_strong_baselines.csv`
- `appendix_reconstruction_ablation.csv`
- `appendix_threshold_sensitivity.csv`
- `appendix_structure_stability.csv`

`table_5_4_strong_baselines.csv` automatically injects the GBSD rows from the
main-blind summary, so the GBSD baseline row and Table 4.2 share the same source.

## Figures

`paper_tools/generate_figures.py` currently generates a figure source manifest,
not final manuscript PNG/PDF/SVG panels:

```powershell
<PYTHON> paper_tools/generate_figures.py
```

The output is:

```text
results/paper_figures/figure_source_manifest.json
```

This manifest maps each paper figure to the official summary CSV that should be
used as source data. Final journal-style plotting and layout should use these
summary files, rather than historical local figures.

## Consistency Checks

Run before using any generated table in the manuscript:

```powershell
<PYTHON> paper_tools/verify_consistency.py
<PYTHON> paper_tools/verify_submission_assets.py
```

The checker verifies that:

- official summaries contain protocol IDs
- paper tables are problem-level mean/std tables, not raw seed-level dumps
- the strong-baseline table contains GBSD rows for all three problems
- GBSD rows in baseline comparisons match the main-blind summary
- guard-ablation tables do not report unrecomputed UQ fields
- official labels do not contain historical `3.x` version names

## Notes on Claims

The official results support a guarded, problem-dependent selection framework:

- Laplace and Poisson usually keep the dense Bayesian student as the final mean
  predictor.
- Burgers inverse often accepts the structured candidate while preserving
  inverse-parameter accuracy.
- Structure-discovery baselines and threshold sweeps are included as diagnostic
  evidence; they should be interpreted under the same blind-test protocol.

Structure stability currently reports outcome-level stability across seeds. It
does not claim symbolic selected-term frequency stability.

## License

This code is released under the MIT License.

Data, configuration files, processed result tables, and figure source outputs
are released under CC BY 4.0; see `DATA_LICENSE.md`.

## Third-Party Provenance

Third-party and historical runtime-code provenance is documented in
`docs/THIRD_PARTY_NOTICE.md`. The GBSD method, guarded source-selection
protocol, blind-test summaries, and paper-table generation are organized by
this repository.

## Citation

Citation metadata are provided in `CITATION.cff`. Author names and DOI are left
as TODO placeholders until the public release or Zenodo archive is finalized.

## Contact and Issues

Please use the GitHub issue tracker for reproducibility questions. If the
repository is released non-anonymously, update this section with the maintainer
contact before publication.
