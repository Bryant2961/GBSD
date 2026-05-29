"""Generate a paper-figure source manifest from official summary files.

The plotting implementation is intentionally conservative at this migration
stage: it creates a figure source manifest and fails if the required summaries
are absent, preventing accidental use of legacy local figures.

This script does not render final manuscript PNG/PDF/SVG panels. Use the
manifest it writes as the audited source-data map for plotting and layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIGURE_SOURCES = {
    "fig_4_5_compression_accuracy": "summary_threshold_sensitivity.csv",
    "fig_5_1_uq_calibration": "summary_uq_ablation.csv",
    "fig_5_2_strong_baseline": "summary_baselines.csv",
    "fig_5_3_structure_baseline": "summary_baselines.csv",
    "appendix_guard_threshold_sensitivity": "summary_threshold_sensitivity.csv",
    "appendix_structure_stability_heatmap": "summary_structure_stability.csv",
    "appendix_reconstruction_ablation": "summary_reconstruction_ablation.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="results/paper_tables")
    parser.add_argument("--output-dir", default="results/paper_figures")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    summary_dir = ROOT / args.summary_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    missing = []
    for figure, source_name in FIGURE_SOURCES.items():
        source = summary_dir / source_name
        if not source.is_file():
            missing.append(source_name)
        manifest.append({"figure": figure, "source_summary_file": source_name})
    with (output_dir / "figure_source_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    if missing and not args.allow_missing:
        print("Missing required summary sources for figures:")
        for item in sorted(set(missing)):
            print(f"  - {item}")
        return 2
    print(f"Wrote figure source manifest to {output_dir}")
    print("Note: final manuscript PNG/PDF/SVG panels are not rendered by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
