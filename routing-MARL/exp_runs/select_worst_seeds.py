from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_experiments import (  # noqa: E402
    count_demand_vehicles,
    summarize_run,
    write_metrics_csv,
    write_simple_yaml,
)


DEFAULT_NETWORK_NAME = "karl"
DEFAULT_ALGORITHM_NAME = "LSTM"
DEFAULT_TOP_K = 10
DEFAULT_RANKING_METRIC = "network_clear_time desc, mean_travel_time desc, median_travel_time desc, seed"


SUMMARY_METRIC_FIELDS = [
    "duration_seconds",
    "demand_vehicles",
    "arrived_vehicles",
    "summary_arrived",
    "summary_running",
    "summary_waiting",
    "summary_mean_speed",
    "num_trips",
    "mean_travel_time",
    "median_travel_time",
    "min_travel_time",
    "max_travel_time",
    "std_travel_time",
    "mean_waiting_time",
    "total_waiting_time",
    "network_clear_time",
    "summary_final_time",
]

SECONDS_TO_MINUTES_FIELDS = {
    "duration_seconds",
    "mean_travel_time",
    "median_travel_time",
    "min_travel_time",
    "max_travel_time",
    "std_travel_time",
    "mean_waiting_time",
    "total_waiting_time",
    "summary_final_time",
}


CSV_FIELDS = [
    "run",
    "run_number",
    "seed",
    "status",
    "reason",
    "returncode",
    "duration_seconds",
    "demand_vehicles",
    "arrived_vehicles",
    "summary_arrived",
    "summary_running",
    "summary_waiting",
    "summary_mean_speed",
    "num_trips",
    "mean_travel_time",
    "median_travel_time",
    "min_travel_time",
    "max_travel_time",
    "std_travel_time",
    "mean_waiting_time",
    "total_waiting_time",
    "network_clear_time",
    "summary_final_time",
    "run_dir",
    "log_file",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the worst complete seeds directly from existing run directories"
    )
    parser.add_argument(
        "network_name",
        nargs="?",
        default=DEFAULT_NETWORK_NAME,
        help=f"Network name under exp_runs/<algorithm>/<network>/ (default: {DEFAULT_NETWORK_NAME})",
    )
    parser.add_argument(
        "algorithm_name",
        nargs="?",
        default=DEFAULT_ALGORITHM_NAME,
        help=f"Algorithm name under exp_runs/<algorithm>/<network>/ (default: {DEFAULT_ALGORITHM_NAME})",
    )
    parser.add_argument(
        "--testcase",
        type=str,
        default=None,
        help="Config/network testcase name. Defaults to network_name.",
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=SCRIPT_DIR,
        help=f"Root folder that contains the exp_runs structure (default: {SCRIPT_DIR})",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of worst complete seeds to select")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the YAML output. Defaults to <exp-root>/<algorithm>/<network>/worst_10_seeds.yaml.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to the same folder as the YAML output with a .csv filename.",
    )
    return parser.parse_args()


def parse_seed_from_run_config(run_dir: Path) -> int | None:
    run_config = run_dir / "run.sumocfg"
    if not run_config.exists():
        return None
    match = re.search(r"<seed\s+value=\"(\d+)\"", run_config.read_text(encoding="utf-8", errors="ignore"))
    return int(match.group(1)) if match else None


def run_number_from_path(run_dir: Path) -> int:
    match = re.fullmatch(r"run_(\d+)", run_dir.name)
    return int(match.group(1)) if match else 0


def build_existing_run_result(run_dir: Path) -> dict[str, Any]:
    seed = parse_seed_from_run_config(run_dir)
    return {
        "run": run_dir.name,
        "run_number": run_number_from_path(run_dir),
        "seed": seed if seed is not None else "",
        "run_dir": str(run_dir),
        "log_file": str(run_dir / "run.log"),
        "command": "",
        "returncode": 0,
        "duration_seconds": "",
        "status": "ran",
        "reason": "",
    }


def discover_run_dirs(base_dir: Path) -> list[Path]:
    run_dirs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"run_\d+", path.name)
    ]
    return sorted(run_dirs, key=run_number_from_path)


def is_complete_seed(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "complete"
        and record.get("arrived_vehicles") == record.get("demand_vehicles")
        and record.get("summary_arrived") == record.get("arrived_vehicles")
    )


def worst_sort_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        -float(record["network_clear_time"]),
        -float(record["mean_travel_time"]),
        -float(record["median_travel_time"]),
        int(record["seed"]),
    )


def select_worst_seeds(records: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], int]:
    complete_records = [
        record
        for record in records
        if is_complete_seed(record)
        and "network_clear_time" in record
        and "mean_travel_time" in record
        and "median_travel_time" in record
    ]
    if len(complete_records) < top_k:
        raise ValueError(f"Need at least {top_k} complete seeds, found {len(complete_records)}.")

    ranked_records = sorted(complete_records, key=worst_sort_key)
    selected_records = []
    for rank, record in enumerate(ranked_records[:top_k], start=1):
        selected_records.append({"rank": rank, **record})
    return selected_records, len(complete_records)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_values(records: list[dict[str, Any]], field_name: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(field_name)
        if value == "" or value is None:
            continue
        if is_number(value):
            values.append(float(value))
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def build_metrics_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_records = [
        record
        for record in records
        if record.get("status") in {"complete", "incomplete"} and "mean_travel_time" in record
    ]
    summary: dict[str, Any] = {
        "total_runs": len(records),
        "parsed_runs": len(parsed_records),
        "complete_runs": sum(1 for record in records if record.get("status") == "complete"),
        "incomplete_runs": sum(1 for record in records if record.get("status") == "incomplete"),
        "failed_runs": sum(1 for record in records if record.get("status") == "failed"),
    }
    for field_name in SUMMARY_METRIC_FIELDS:
        values = numeric_values(parsed_records, field_name)
        if not values:
            continue
        summary[field_name] = statistics.mean(values)
        summary[f"{field_name}_std_across_runs"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return summary


def convert_summary_to_minutes(summary: dict[str, Any]) -> dict[str, Any]:
    converted = dict(summary)
    for field_name in SECONDS_TO_MINUTES_FIELDS:
        for key in (field_name, f"{field_name}_std_across_runs"):
            value = converted.get(key)
            if is_number(value):
                converted[key] = value / 60.0
    converted["time_unit"] = "minutes_except_network_clear_time_seconds"
    return converted


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_yaml_payload(
    selected_records: list[dict[str, Any]],
    complete_seed_count: int,
    top_k: int,
    worst_seed_summary: dict[str, Any],
    all_runs_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ranking_metric": DEFAULT_RANKING_METRIC,
        "selection_top_k": top_k,
        "complete_seed_count": complete_seed_count,
        "summary_scope": "selected_worst_seeds",
        "summary": worst_seed_summary,
        "all_runs_summary": all_runs_summary,
        "worst_seeds": [
            {key: record[key] for key in ["rank", *CSV_FIELDS]}
            for record in selected_records
        ],
    }


def resolve_output_path(base_dir: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return base_dir / "worst_10_seeds.yaml"


def resolve_csv_output_path(output_path: Path, csv_output: Path | None) -> Path:
    if csv_output is not None:
        return csv_output
    return output_path.with_suffix(".csv")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["rank", *CSV_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    exp_root = args.exp_root.resolve()
    base_dir = exp_root / args.algorithm_name / args.network_name
    if not base_dir.exists():
        raise FileNotFoundError(f"Could not find experiment directory: {base_dir}")

    run_dirs = discover_run_dirs(base_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base_dir}")

    testcase = args.testcase or args.network_name
    demand_count = count_demand_vehicles(testcase, args.network_name)
    records = [summarize_run(build_existing_run_result(run_dir), demand_count) for run_dir in run_dirs]
    selected_records, complete_seed_count = select_worst_seeds(records, args.top_k)
    all_runs_summary = build_metrics_summary(records)
    worst_seed_summary = build_metrics_summary(selected_records)
    all_runs_summary_minutes = convert_summary_to_minutes(all_runs_summary)
    worst_seed_summary_minutes = convert_summary_to_minutes(worst_seed_summary)

    metrics_path = base_dir / "metrics_by_seed.csv"
    summary_path = base_dir / "metrics_summary.json"
    summary_minutes_path = base_dir / "metrics_summary_minutes.json"
    output_path = resolve_output_path(base_dir, args.output).resolve()
    csv_output_path = resolve_csv_output_path(output_path, args.csv_output).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    write_metrics_csv(metrics_path, records)
    write_json(
        summary_path,
        {
            "base_output_dir": str(base_dir),
            "summary_scope": "selected_worst_seeds",
            "aggregate": worst_seed_summary,
            "all_runs_aggregate": all_runs_summary,
            "runs": records,
            "worst_seeds": selected_records,
        },
    )
    write_json(
        summary_minutes_path,
        {
            "base_output_dir": str(base_dir),
            "summary_scope": "selected_worst_seeds",
            "aggregate": worst_seed_summary_minutes,
            "all_runs_aggregate": all_runs_summary_minutes,
        },
    )
    write_simple_yaml(
        output_path,
        build_yaml_payload(
            selected_records,
            complete_seed_count,
            args.top_k,
            worst_seed_summary_minutes,
            all_runs_summary_minutes,
        ),
    )
    write_csv(csv_output_path, selected_records)

    selected_seeds = ", ".join(str(record["seed"]) for record in selected_records)
    print(f"Scanned {len(run_dirs)} run directories.")
    print(f"Selected {len(selected_records)} worst complete seeds from {complete_seed_count} complete rows.")
    print(f"Worst seeds: {selected_seeds}")
    print(f"Wrote metrics CSV to {metrics_path}")
    print(f"Wrote metrics summary to {summary_path}")
    print(f"Wrote minute metrics summary to {summary_minutes_path}")
    print(f"Wrote YAML to {output_path}")
    print(f"Wrote CSV to {csv_output_path}")


if __name__ == "__main__":
    main()
