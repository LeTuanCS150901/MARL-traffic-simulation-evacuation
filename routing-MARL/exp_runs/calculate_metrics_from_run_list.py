from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Manually edit this list to control which run numbers are included.
# rander
# RUN_NUMBER_LIST: list[int] = [1072, 1051, 1032, 1020, 1046, 1088, 1017, 1028, 1070, 1105]
# manhattan2
# RUN_NUMBER_LIST: list[int] = [83, 66, 42, 57, 63, 50, 18, 29, 72, 47]

# KARL
RUN_NUMBER_LIST: list[int] = [62, 98, 54, 36, 51, 55, 59, 38, 47, 50]


from run_experiments import (  # noqa: E402
    DEFAULT_TOP_K,
    aggregate_metrics,
    count_demand_vehicles,
    sort_records,
    summarize_run,
    write_best_seeds,
    write_metrics_csv,
)


def parse_seed_from_run_config(run_dir: Path) -> int | None:
    run_config = run_dir / "run.sumocfg"
    if not run_config.exists():
        return None
    match = re.search(r"<seed\s+value=\"(\d+)\"", run_config.read_text(encoding="utf-8", errors="ignore"))
    return int(match.group(1)) if match else None


def build_existing_run_result(run_dir: Path) -> dict:
    match = re.fullmatch(r"run_(\d+)", run_dir.name)
    run_number = int(match.group(1)) if match else 0
    seed = parse_seed_from_run_config(run_dir)
    return {
        "run": run_dir.name,
        "run_number": run_number,
        "seed": seed if seed is not None else "",
        "run_dir": str(run_dir),
        "log_file": str(run_dir / "run.log"),
        "command": "",
        "returncode": 0,
        "duration_seconds": "",
        "status": "ran",
        "reason": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate seed-search metrics from a manually specified list of run numbers"
        )
    )
    parser.add_argument("network_name", type=str, help="Network name under exp_runs/<algorithm>/<network>/")
    parser.add_argument("algorithm_name", type=str, help="Algorithm name under exp_runs/<algorithm>/<network>/")
    parser.add_argument(
        "--testcase",
        type=str,
        default=None,
        help="Config/network testcase name. Defaults to network_name.",
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        default=str(SCRIPT_DIR),
        help="Root folder that contains the exp_runs structure",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--strict-run-list",
        action="store_true",
        help="Fail if RUN_NUMBER_LIST yields no matching run directories.",
    )
    args = parser.parse_args()

    exp_root = Path(args.exp_root).resolve()
    base_dir = exp_root / args.algorithm_name / args.network_name
    if not base_dir.exists():
        raise FileNotFoundError(f"Could not find experiment directory: {base_dir}")

    run_numbers = list(RUN_NUMBER_LIST)
    if not run_numbers:
        run_numbers = []

    testcase = args.testcase or args.network_name
    demand_count = count_demand_vehicles(testcase, args.network_name)

    all_run_dirs = sorted(
        [
            path
            for path in base_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("run_")
            and path.name.split("_", 1)[1].isdigit()
        ],
        key=lambda path: int(path.name.split("_", 1)[1]),
    )

    run_dirs = all_run_dirs
    if run_numbers:
        run_number_set = set(run_numbers)
        run_dirs = [
            path for path in all_run_dirs if int(path.name.split("_", 1)[1]) in run_number_set
        ]

    if not run_dirs:
        if args.strict_run_list or not all_run_dirs:
            raise FileNotFoundError(f"No matching run directories found under {base_dir}")
        print(
            "Warning: RUN_NUMBER_LIST did not match any runs. Falling back to all runs "
            f"under {base_dir}."
        )
        run_dirs = all_run_dirs

    records = [summarize_run(build_existing_run_result(run_dir), demand_count) for run_dir in run_dirs]
    sorted_records = sort_records(records, "mean_travel_time")
    aggregate = aggregate_metrics(records)

    csv_path = base_dir / "metrics_by_seed.csv"
    legacy_csv_path = base_dir / "metrics_by_run.csv"
    json_path = base_dir / "metrics_summary.json"
    best_seeds_path = base_dir / "best_seeds.yaml"

    write_metrics_csv(csv_path, records)
    with legacy_csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        fieldnames = ["run", "seed", "status", "arrived_vehicles", "mean_travel_time", "mean_waiting_time"]
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write_best_seeds(best_seeds_path, sorted_records, args.top_k, "mean_travel_time")

    payload = {
        "base_output_dir": str(base_dir),
        "selection_top_k": args.top_k,
        "demand_vehicles": demand_count,
        "aggregate": aggregate,
        "runs": records,
        "best_seeds": sorted_records[: args.top_k],
        "run_numbers": run_numbers,
    }
    with json_path.open("w", encoding="utf-8") as json_handle:
        json.dump(payload, json_handle, indent=2)

    print(f"Saved per-seed metrics to {csv_path}")
    print(f"Saved legacy run metrics to {legacy_csv_path}")
    print(f"Saved summary metrics to {json_path}")
    print(f"Saved best seeds to {best_seeds_path}")


if __name__ == "__main__":
    main()
