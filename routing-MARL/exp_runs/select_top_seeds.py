from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_PATH = SCRIPT_DIR / "proposed" / "karl" / "metrics_by_seed.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "proposed" / "karl" / "top_10_seeds.yaml"
DEFAULT_RANKING_METRIC = "mean_travel_time, -arrived_vehicles, mean_waiting_time, seed"


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


INT_FIELDS = {
    "run_number",
    "seed",
    "returncode",
    "demand_vehicles",
    "arrived_vehicles",
    "summary_arrived",
    "summary_running",
    "summary_waiting",
    "num_trips",
    "network_clear_time",
    "summary_final_time",
}
FLOAT_FIELDS = {
    "duration_seconds",
    "summary_mean_speed",
    "mean_travel_time",
    "median_travel_time",
    "min_travel_time",
    "max_travel_time",
    "std_travel_time",
    "mean_waiting_time",
    "total_waiting_time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the top complete seeds from a metrics_by_seed.csv file")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Path to metrics_by_seed.csv (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of complete seeds to select")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the YAML output (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to the same folder as the YAML output with a .csv filename.",
    )
    return parser.parse_args()


def parse_int(value: str, field_name: str) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {field_name}: {value!r}") from exc


def parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from exc


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(row)
    for field_name in INT_FIELDS:
        if field_name in normalized and normalized[field_name] != "":
            normalized[field_name] = parse_int(normalized[field_name], field_name)
    for field_name in FLOAT_FIELDS:
        if field_name in normalized and normalized[field_name] != "":
            normalized[field_name] = parse_float(normalized[field_name], field_name)
    return normalized


def is_complete_seed(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "complete"
        and row.get("arrived_vehicles") == row.get("demand_vehicles")
        and row.get("summary_arrived") == row.get("arrived_vehicles")
    )


def sort_key(row: dict[str, Any]) -> tuple[float, int, float, int]:
    return (
        float(row["mean_travel_time"]),
        -int(row["arrived_vehicles"]),
        float(row["mean_waiting_time"]),
        int(row["seed"]),
    )


def yaml_quote(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append("  -")
                    for child_key, child_value in item.items():
                        lines.append(f"    {child_key}: {yaml_quote(child_value)}")
                else:
                    lines.append(f"  - {yaml_quote(item)}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {yaml_quote(child_value)}")
        else:
            lines.append(f"{key}: {yaml_quote(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["rank", *CSV_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_top_seeds(metrics_path: Path, top_k: int) -> tuple[list[dict[str, Any]], int]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [normalize_row(row) for row in reader]

    complete_rows = [row for row in rows if is_complete_seed(row)]
    if len(complete_rows) < top_k:
        raise ValueError(
            f"Need at least {top_k} complete seeds in {metrics_path}, found {len(complete_rows)}."
        )

    ranked_rows = sorted(complete_rows, key=sort_key)
    selected_rows = []
    for rank, row in enumerate(ranked_rows[:top_k], start=1):
        selected_rows.append({"rank": rank, **row})
    return selected_rows, len(complete_rows)


def build_yaml_payload(selected_rows: list[dict[str, Any]], complete_seed_count: int, top_k: int) -> dict[str, Any]:
    return {
        "ranking_metric": DEFAULT_RANKING_METRIC,
        "selection_top_k": top_k,
        "complete_seed_count": complete_seed_count,
        "top_seeds": [
            {key: row[key] for key in ["rank", *CSV_FIELDS]}
            for row in selected_rows
        ],
    }


def resolve_csv_output_path(output_path: Path, csv_output: Path | None) -> Path:
    if csv_output is not None:
        return csv_output
    return output_path.with_suffix(".csv")


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics.resolve()
    output_path = args.output.resolve()
    csv_output_path = resolve_csv_output_path(output_path, args.csv_output)

    selected_rows, complete_seed_count = select_top_seeds(metrics_path, args.top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_payload = build_yaml_payload(selected_rows, complete_seed_count, args.top_k)
    write_yaml(output_path, yaml_payload)
    write_csv(csv_output_path, selected_rows)

    selected_seeds = ", ".join(str(row["seed"]) for row in selected_rows)
    print(f"Selected {len(selected_rows)} complete seeds from {complete_seed_count} complete rows.")
    print(f"Top seeds: {selected_seeds}")
    print(f"Wrote YAML to {output_path}")
    print(f"Wrote CSV to {csv_output_path}")


if __name__ == "__main__":
    main()
