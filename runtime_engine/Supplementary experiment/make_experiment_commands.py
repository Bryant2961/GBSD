"""Generate PowerShell command files for supplementary training ablations."""
from __future__ import annotations

from pathlib import Path

from common import COMMAND_DIR, PROJECT_ROOT, ensure_output_dirs


PY = "python"
ROOT = str(PROJECT_ROOT)


def header(title: str) -> list[str]:
    return [
        f"# {title}",
        "# Self-contained runnable package command.",
        "# It changes only this package's working Results directory.",
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location -LiteralPath {ROOT!r}",
        "",
    ]


def write_script(name: str, lines: list[str]) -> None:
    COMMAND_DIR.mkdir(parents=True, exist_ok=True)
    path = COMMAND_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


def cmd(script: str, *args: str) -> str:
    joined = " ".join(args)
    return f'& {PY} ".\\Supplementary experiment\\{script}" {joined}'.rstrip()


def run_all(*args: str) -> str:
    return f"& {PY} .\\run_all_experiments.py {' '.join(args)}"


def main() -> None:
    ensure_output_dirs()

    seed_lines = header("P0 Bayesian guarded multi-seed sweep")
    for seed in [0, 1, 2, 3, 4]:
        seed_lines.append(run_all(
            "--case all --method bayesian --preset full",
            f"--seed {seed} --clean",
        ))
        seed_lines.append("")
    write_script("p0_seed_sweep.ps1", seed_lines)

    post_lines = header("P0 post-processing tables")
    post_lines += [
        cmd("aggregate_multiseed.py"),
        "",
        cmd("guard_ablation.py"),
        "",
        cmd("uq_ablation.py", "--case Poisson"),
        "",
        cmd("runtime_and_params.py"),
        "",
        cmd("reconstruction_ablation_summary.py"),
        "",
        cmd("clustering_sweep_summary.py"),
        "",
    ]
    write_script("p0_postprocess_tables.ps1", post_lines)

    direct_lines = header("P1 direct MC Dropout PINN-style baseline")
    direct_lines += [
        "# This is a zero-distillation MC Dropout student ablation, not a separate pipeline.",
        "# It disables student distillation weights while retaining physics-informed student losses.",
        "# Final source is forced to dense by setting an unreachable compression threshold.",
        "",
    ]
    for case in ["Laplace", "Poisson", "Burgers_inv"]:
        direct_lines.append(cmd(
            "run_config_ablation.py",
            f"--case {case} --tag direct_mc_dropout_pinn_{case} --seed 0",
            "--method bayesian --preset full --clean",
            "--set lambda_distill_student=0",
            "--set lambda_distill_mean_refine=0",
            "--set lambda_distill_recon=0",
            "--set force_structured_prediction=0",
            "--set min_structured_compression=999",
        ))
        direct_lines.append("")
    write_script("p1_direct_mc_dropout_pinn.ps1", direct_lines)

    recon_lines = header("P0 structured reconstruction ablations")
    for case in ["Poisson", "Burgers_inv"]:
        recon_lines.append(cmd(
            "run_config_ablation.py",
            f"--case {case} --tag no_dense_anchor_{case} --seed 0",
            "--method bayesian --preset full --clean",
            "--set lambda_anchor_recon=0",
            "--set anchor_pretrain_steps=0",
        ))
        recon_lines.append(cmd(
            "run_config_ablation.py",
            f"--case {case} --tag no_pde_reconstruction_{case} --seed 0",
            "--method bayesian --preset full --clean",
            "--set lambda_pde_recon=0",
        ))
        recon_lines.append("")
    write_script("p0_reconstruction_ablations.ps1", recon_lines)

    sweep_lines = header("P2 clustering threshold sweep")
    for case in ["Poisson", "Burgers_inv"]:
        for dist in ["0.03", "0.05", "0.08", "0.10", "0.15", "0.20"]:
            tag = f"cluster_{case}_{dist.replace('.', 'p')}"
            sweep_lines.append(cmd(
                "run_config_ablation.py",
                f"--case {case} --tag {tag} --seed 0",
                "--method bayesian --preset full --clean",
                f"--set cluster_distance={dist}",
            ))
        sweep_lines.append("")
    write_script("p2_clustering_sweep.ps1", sweep_lines)

    print("\nGenerated supplementary command files.")


if __name__ == "__main__":
    main()
