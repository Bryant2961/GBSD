# Repository Checklist

Last local check: 2026-05-29

## Structure

- [x] Root README exists.
- [x] Reproduction guide exists.
- [x] MIT license file exists.
- [x] Data/result license file exists.
- [x] Citation metadata file exists.
- [x] Source code is organized under `src/`, `experiments/`, `paper_tools/`,
      `postprocess/`, `runtime_engine/`, and `scripts/`.
- [x] Historical provenance is documented under `legacy/`.

## Experiment Scripts

- [x] Main experiment runner exists.
- [x] Strong-baseline runner exists.
- [x] Guard/UQ/reconstruction ablation wrappers exist.
- [x] Structure-diagnostics wrapper exists.
- [x] Single-problem convenience runners exist.
- [x] Smoke-test script exists.

## Generated Tables and Assets

- [x] `paper_tools/generate_tables.py` ran successfully.
- [x] `paper_tools/generate_figures.py` ran successfully as a figure-source
      manifest generator.
- [x] `paper_tools/verify_consistency.py` passed.
- [x] `paper_tools/verify_submission_assets.py` passed.
- [x] `paper_tools/export_submission_assets.py` ran successfully.

## Tests

- [ ] `python -m pytest tests -q` requires `pytest` to be installed in the
      active environment.
- [x] The small test functions were manually executed in the local environment.

## Known Limits

- [ ] `CITATION.cff` still contains TODO author and DOI fields until the final
      public release or Zenodo archive is prepared.
- [ ] `paper_tools/generate_figures.py` creates a source manifest, not final
      manuscript PNG/PDF/SVG panels.
- [ ] Full per-seed outputs and checkpoints are not included in GitHub and
      should be regenerated or archived separately.
- [ ] Structure-stability summaries are outcome-level stability diagnostics,
      not symbolic selected-term frequency tables.

## Upload Safety

- [x] Python cache files were removed from the release folder.
- [x] `.gitignore` excludes local caches, checkpoints, raw binary arrays,
      generated runtime outputs, Word temporary files, and tracking folders.
- [x] Generated paper tables contain no local absolute paths or historical
      version tags in the manuscript-ready CSV files.
- [ ] Replace `Anonymous Authors` in `LICENSE` if the repository is no longer
      anonymous.

