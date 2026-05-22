from __future__ import annotations

from pathlib import Path
import argparse
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.benchmark import SELECTED_CODES, JSSPInstance, load_selected_instances
from src.jssp import decode_sequence, optimality_gap, save_schedule, schedule_to_dataframe
from src.reporting import save_dataframe
from src.sa import SAConfig, run_sa


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Job Shop Scheduling experiments with Simulated Annealing.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tune-runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2026)
    return parser.parse_args()


def tune_parameters(
    instances: dict[str, JSSPInstance], output_dir: Path, tune_runs: int, base_seed: int
) -> tuple[SAConfig, pd.DataFrame]:
    grid = [
        {"iterations": 2500, "initial_temperature_ratio": 0.20, "cooling_rate": 0.995},
        {"iterations": 3500, "initial_temperature_ratio": 0.25, "cooling_rate": 0.996},
        {"iterations": 4500, "initial_temperature_ratio": 0.30, "cooling_rate": 0.996},
        {"iterations": 5500, "initial_temperature_ratio": 0.35, "cooling_rate": 0.997},
        {"iterations": 6500, "initial_temperature_ratio": 0.40, "cooling_rate": 0.997},
        {"iterations": 5000, "initial_temperature_ratio": 0.25, "cooling_rate": 0.998},
    ]
    tuning_instances = ["ft10", "la16"]
    records: list[dict[str, float | int | str]] = []
    for index, params in enumerate(grid, start=1):
        gaps = []
        runtimes = []
        acceptance_rates = []
        for instance_code in tuning_instances:
            instance = instances[instance_code]
            for run_idx in range(tune_runs):
                config = SAConfig(seed=base_seed + index * 100 + run_idx, **params)
                result = run_sa(instance, config)
                gaps.append(optimality_gap(result.best_makespan, instance.optimum))
                runtimes.append(result.runtime_seconds)
                acceptance_rates.append(result.accepted_moves / result.iterations)
        mean_gap = float(np.mean(gaps))
        mean_runtime = float(np.mean(runtimes))
        records.append(
            {
                "config_id": f"C{index}",
                **params,
                "mean_gap_percent": mean_gap,
                "std_gap_percent": float(np.std(gaps, ddof=0)),
                "mean_runtime_seconds": mean_runtime,
                "mean_acceptance_rate": float(np.mean(acceptance_rates)),
                "score": mean_gap + 0.03 * mean_runtime,
            }
        )

    tuning_df = pd.DataFrame(records).sort_values(["score", "mean_gap_percent", "mean_runtime_seconds"])
    save_dataframe(tuning_df, output_dir / "tables" / "parameter_tuning.csv")
    best = tuning_df.iloc[0]
    return (
        SAConfig(
            iterations=int(best["iterations"]),
            initial_temperature_ratio=float(best["initial_temperature_ratio"]),
            cooling_rate=float(best["cooling_rate"]),
            seed=base_seed,
            record_interval=50,
        ),
        tuning_df,
    )


def run_full_experiment(
    instances: dict[str, JSSPInstance], config: SAConfig, runs: int, base_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    run_records = []
    convergence_records = []
    best_schedules = {}
    baseline_schedules = {}

    for instance_code in SELECTED_CODES:
        instance = instances[instance_code]
        best_result = None
        baseline_schedule = None
        for run_idx in range(1, runs + 1):
            seed = base_seed + 1000 * (list(SELECTED_CODES).index(instance_code) + 1) + run_idx
            run_config = SAConfig(
                iterations=config.iterations,
                initial_temperature_ratio=config.initial_temperature_ratio,
                cooling_rate=config.cooling_rate,
                seed=seed,
                record_interval=config.record_interval,
            )
            result = run_sa(instance, run_config)
            baseline_schedule = result.baseline_schedule
            run_records.append(
                {
                    "instance": instance.code,
                    "difficulty": instance.difficulty,
                    "jobs": instance.jobs,
                    "machines": instance.machines,
                    "operations": instance.operation_count,
                    "optimum": instance.optimum,
                    "baseline_spt": result.baseline_makespan,
                    "baseline_gap_percent": optimality_gap(result.baseline_makespan, instance.optimum),
                    "run": run_idx,
                    "seed": seed,
                    "initial_makespan": result.initial_makespan,
                    "best_found": result.best_makespan,
                    "gap_percent": optimality_gap(result.best_makespan, instance.optimum),
                    "accepted_moves": result.accepted_moves,
                    "improving_moves": result.improving_moves,
                    "acceptance_rate": result.accepted_moves / result.iterations,
                    "iterations": result.iterations,
                    "runtime_seconds": result.runtime_seconds,
                }
            )
            for row in result.history:
                convergence_records.append(
                    {
                        "instance": instance.code,
                        "run": run_idx,
                        **row,
                    }
                )
            if best_result is None or result.best_makespan < best_result.best_makespan:
                best_result = result
        if best_result is None or baseline_schedule is None:
            raise RuntimeError(f"No SA results were produced for {instance_code}")
        best_schedules[instance_code] = best_result.best_schedule
        baseline_schedules[instance_code] = baseline_schedule

    return pd.DataFrame(run_records), pd.DataFrame(convergence_records), best_schedules, baseline_schedules


def summarise_results(run_df: pd.DataFrame) -> pd.DataFrame:
    return (
        run_df.groupby(
            [
                "instance",
                "difficulty",
                "jobs",
                "machines",
                "operations",
                "optimum",
                "baseline_spt",
                "baseline_gap_percent",
            ],
            as_index=False,
        )
        .agg(
            best_found=("best_found", "min"),
            mean_found=("best_found", "mean"),
            std_found=("best_found", "std"),
            worst_found=("best_found", "max"),
            mean_gap_percent=("gap_percent", "mean"),
            best_gap_percent=("gap_percent", "min"),
            mean_acceptance_rate=("acceptance_rate", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
        )
        .sort_values("instance")
    )


def plot_convergence(convergence_df: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for axis, instance_code in zip(axes, list(SELECTED_CODES.keys()) + [None]):
        if instance_code is None:
            axis.axis("off")
            continue
        subset = convergence_df[convergence_df["instance"] == instance_code]
        averaged = subset.groupby("iteration", as_index=False)["best_makespan"].mean()
        axis.plot(averaged["iteration"], averaged["best_makespan"], color="#006d77", linewidth=2.4)
        axis.set_title(f"{instance_code.upper()} SA Convergence")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Best Makespan")
    fig.suptitle("Simulated Annealing Convergence Across Job Shop Benchmarks", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "figures" / "convergence_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.36
    ax.bar(x - width / 2, summary_df["baseline_gap_percent"], width=width, label="SPT Baseline", color="#adb5bd")
    ax.bar(x + width / 2, summary_df["mean_gap_percent"], width=width, label="SA Mean", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([row.instance.upper() for row in summary_df.itertuples()])
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_title("Average Optimality Gap: SPT Baseline vs Simulated Annealing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "gap_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_scaling(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{row.instance.upper()}\n{row.jobs}x{row.machines}" for row in summary_df.itertuples()]
    ax.bar(labels, summary_df["mean_runtime_seconds"], color="#264653")
    ax.set_title("Average Runtime per Benchmark Instance")
    ax.set_ylabel("Seconds")
    for idx, value in enumerate(summary_df["mean_runtime_seconds"]):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "runtime_scaling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_acceptance(convergence_df: pd.DataFrame, output_dir: Path) -> None:
    sample = convergence_df[(convergence_df["instance"] == "ft10") & (convergence_df["run"] == 1)].copy()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sample["iteration"], sample["temperature"], color="#e76f51", linewidth=2.3, label="Temperature")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Temperature", color="#e76f51")
    ax2 = ax1.twinx()
    ax2.plot(sample["iteration"], sample["best_makespan"], color="#006d77", linewidth=2.3, label="Best makespan")
    ax2.set_ylabel("Best Makespan", color="#006d77")
    ax1.set_title("Cooling Schedule and Search Progress (FT10, Run 1)")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "temperature_progress.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gantt(instance: JSSPInstance, schedule, title: str, output_path: Path) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(15, 6))
    cmap = plt.get_cmap("tab20", instance.jobs)
    for op in schedule.operations:
        ax.barh(
            y=op.machine,
            width=op.duration,
            left=op.start,
            height=0.76,
            color=cmap(op.job % 20),
            edgecolor="white",
            linewidth=0.7,
        )
        if instance.jobs <= 10 and op.duration >= 3:
            ax.text(op.start + op.duration / 2, op.machine, f"J{op.job + 1}", ha="center", va="center", fontsize=8)
    ax.set_yticks(range(instance.machines))
    ax.set_yticklabels([f"M{machine + 1}" for machine in range(instance.machines)])
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_title(title)
    handles = [Patch(facecolor=cmap(job % 20), label=f"J{job + 1}") for job in range(min(instance.jobs, 10))]
    ax.legend(handles=handles, ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metadata(
    output_dir: Path,
    config: SAConfig,
    selected_instances: dict[str, JSSPInstance],
    summary_df: pd.DataFrame,
) -> None:
    payload = {
        "problem": "Job Shop Scheduling Problem",
        "algorithm": "Simulated Annealing",
        "objective": "Minimize makespan (Cmax)",
        "selected_instances": [
            {
                "code": code,
                "difficulty": instance.difficulty,
                "jobs": instance.jobs,
                "machines": instance.machines,
                "operations": instance.operation_count,
                "optimum": instance.optimum,
                "source": instance.source,
            }
            for code, instance in selected_instances.items()
        ],
        "final_config": {
            "iterations": config.iterations,
            "initial_temperature_ratio": config.initial_temperature_ratio,
            "cooling_rate": config.cooling_rate,
            "record_interval": config.record_interval,
        },
        "summary_rows": summary_df.to_dict(orient="records"),
    }
    (output_dir / "tables" / "experiment_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_source_notes(output_dir: Path) -> None:
    notes = """Benchmark source notes

Primary benchmark source: OR-Library Job Shop Scheduling data set.
Downloaded raw data file: data/raw/jobshop1.txt
Selected instance files and optimum metadata: JSPLIB mirror of OR-Library benchmark instances.
Selected instances: FT06, LA01, FT10, LA16, LA21.

All reported tables and figures in this project are generated from local Python code and the benchmark files included in data/raw/.
"""
    (output_dir / "tables" / "source_notes.txt").write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = _resolve(args.data_dir)
    output_dir = _resolve(args.output_dir)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    instances = load_selected_instances(data_dir)
    selected_instances = {code: instances[code] for code in SELECTED_CODES}

    best_config, tuning_df = tune_parameters(selected_instances, output_dir, args.tune_runs, args.base_seed)
    run_df, convergence_df, best_schedules, baseline_schedules = run_full_experiment(
        selected_instances, best_config, args.runs, args.base_seed
    )
    summary_df = summarise_results(run_df)

    save_dataframe(run_df, output_dir / "tables" / "run_level_results.csv")
    save_dataframe(convergence_df, output_dir / "tables" / "convergence_history.csv")
    save_dataframe(summary_df, output_dir / "tables" / "summary_results.csv")
    save_metadata(output_dir, best_config, selected_instances, summary_df)
    write_source_notes(output_dir)

    schedule_rows = []
    for code, schedule in best_schedules.items():
        df = schedule_to_dataframe(schedule)
        df.insert(0, "instance", code)
        schedule_rows.append(df)
        save_schedule(schedule, output_dir / "tables" / f"best_schedule_{code}.csv")
    save_dataframe(pd.concat(schedule_rows, ignore_index=True), output_dir / "tables" / "best_schedules.csv")

    plot_convergence(convergence_df, output_dir)
    plot_gap_comparison(summary_df, output_dir)
    plot_runtime_scaling(summary_df, output_dir)
    plot_temperature_acceptance(convergence_df, output_dir)
    plot_gantt(
        selected_instances["ft06"],
        best_schedules["ft06"],
        "Best SA Schedule Gantt Chart (FT06 / 6x6)",
        output_dir / "figures" / "gantt_ft06_sa.png",
    )
    plot_gantt(
        selected_instances["ft06"],
        baseline_schedules["ft06"],
        "SPT Baseline Gantt Chart (FT06 / 6x6)",
        output_dir / "figures" / "gantt_ft06_spt.png",
    )

    print("Best configuration selected:")
    print(tuning_df.iloc[0].to_string())
    print("\nSummary results:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
