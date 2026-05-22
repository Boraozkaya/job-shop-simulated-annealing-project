from __future__ import annotations

from pathlib import Path
import argparse

from src.benchmark import load_selected_instances
from src.jssp import optimality_gap
from src.sa import SAConfig, run_sa


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live demo runner for Job Shop Scheduling with SA.")
    parser.add_argument("--instance", default="ft10")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--temperature-ratio", type=float, default=0.30)
    parser.add_argument("--cooling-rate", type=float, default=0.997)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    instances = load_selected_instances(data_dir)
    instance = instances[args.instance.lower()]
    config = SAConfig(
        seed=args.seed,
        iterations=args.iterations,
        initial_temperature_ratio=args.temperature_ratio,
        cooling_rate=args.cooling_rate,
        record_interval=max(25, args.iterations // 25),
    )
    result = run_sa(instance, config)
    print(f"Instance: {instance.code.upper()} ({instance.jobs} jobs x {instance.machines} machines)")
    print(f"Known optimum/reference: {instance.optimum}")
    print(f"SPT baseline: {result.baseline_makespan}")
    print(f"Initial solution: {result.initial_makespan}")
    print(f"Best found by SA: {result.best_makespan}")
    print(f"Best gap: {optimality_gap(result.best_makespan, instance.optimum):.2f}%")
    print(f"Accepted moves: {result.accepted_moves} / {result.iterations}")
    print(f"Runtime: {result.runtime_seconds:.3f} seconds")
    print("Convergence history:")
    for row in result.history:
        print(
            f"  Iteration {int(row['iteration']):04d}: "
            f"best={int(row['best_makespan'])}, "
            f"current={int(row['current_makespan'])}, "
            f"temperature={float(row['temperature']):.3f}"
        )


if __name__ == "__main__":
    main()
