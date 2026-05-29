from _run_common import run_stage


if __name__ == "__main__":
    raise SystemExit(run_stage(
        "ablations",
        description="Run official ablation stage, including guard ablation outputs.",
    ))

