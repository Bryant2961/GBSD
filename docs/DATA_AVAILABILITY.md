# Data Availability

This repository includes the code, configuration files, small benchmark data
files required by the migrated runtime engine, and aggregated paper-result CSV
files needed to reproduce the reported tables.

## Included in GitHub

- Source code under `src/`, `experiments/`, `paper_tools/`, `postprocess/`, and
  `runtime_engine/`
- Configuration files under `configs/`
- Small benchmark reference data needed by the runtime engine
- Aggregated summary CSV files under `results/paper_tables/`
- Generated paper-table CSV files under `results/paper_tables/generated/`
- Figure source manifest under `results/paper_figures/`

## Not Included in GitHub

The following generated artifacts are intentionally excluded from Git:

- full per-seed official result folders under `results/unified_blind_protocol/`
- runtime work products under `runtime_engine/Results/` and
  `runtime_engine/results/`
- model checkpoints (`*.pth`, `*.pt`, `*.ckpt`)
- raw prediction arrays unless explicitly required as small reference data
- smoke-test outputs and local render caches

These artifacts should be regenerated from the provided scripts or archived as
external release assets.

## Recommended External Archive

For formal submission or publication, archive the complete reproducibility
bundle on Zenodo or a comparable research-data repository. The archive should
include:

- the GitHub source snapshot
- full official per-seed outputs
- checkpoints if required for fast verification
- generated figures if not produced directly by this repository
- environment metadata

Do not invent a DOI before the archive is created. Update `CITATION.cff` and
the manuscript data-availability statement after the DOI is assigned.

## Licenses

Source code is released under the MIT License. Configuration files, processed
CSV results, and figure source-data outputs are released under CC BY 4.0; see
`DATA_LICENSE.md`.

