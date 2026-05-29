# Zenodo Release Notes Draft

Recommended release name:

```text
v1.0.0-jcp-submission
```

Recommended title:

```text
Guarded Bayesian Structured Distillation for PDE Learning: Reproducibility Package
```

Recommended description:

```text
This archive contains the source code, configuration files, official summary
tables, figure source manifest, and reproducibility scripts for the manuscript
"Guarded Bayesian Structured Distillation for PDE Learning". The package
supports the main Laplace, Poisson, and Burgers inverse experiments, strong
baselines, guard ablation, UQ calibration ablation, structured reconstruction
diagnostics, threshold sensitivity, and consistency checks under a unified
five-seed blind-test protocol.
```

Recommended keywords:

- physics-informed neural networks
- Bayesian neural networks
- uncertainty quantification
- model compression
- structure discovery
- partial differential equations
- inverse problems
- reproducibility

Licenses:

- source code: MIT
- data, configurations, aggregated results, and figure source outputs: CC BY 4.0

DOI:

```text
TODO until the Zenodo release is generated.
```

Pre-release checklist:

1. Replace anonymous or TODO author metadata if the repository is public.
2. Run `python paper_tools/generate_tables.py`.
3. Run `python paper_tools/generate_figures.py`.
4. Run `python paper_tools/verify_consistency.py`.
5. Run `python paper_tools/verify_submission_assets.py`.
6. Confirm that no private local paths, API keys, Word temporary files, or
   personal identifiers are included.
7. Decide whether full per-seed outputs and checkpoints should be uploaded as
   Zenodo artifacts rather than GitHub files.
8. After Zenodo assigns a DOI, update `CITATION.cff`, `README.md`, and the
   manuscript data-availability statement.

