from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNTIME_DIR = PROJECT_ROOT / "routing_MARl"
MAIN_SCRIPT = RUNTIME_DIR / "main.py"
DEFAULT_TOP_K = 5
SUMMARY_TRIPINFO_ARRIVAL_TOLERANCE = 0


class ConfigError(ValueError):
    pass


class RunMetricsError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedJob:
    run_number: int
    seed: int
    run_output_dir: Path
    command: list[str]


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def simple_yaml_load(config_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    lines = config_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = strip_inline_comment(lines[index]).rstrip()
        index += 1
        if not raw_line.strip():
            continue
        if raw_line.startswith((" ", "\t")):
            continue
        if ":" not in raw_line:
            raise ConfigError(f"Cannot parse YAML line in {config_path}: {raw_line}")
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            config[key] = parse_scalar(value)
            continue

        items = []
        while index < len(lines):
            child = strip_inline_comment(lines[index]).rstrip()
            if not child.strip():
                index += 1
                continue
            if not child.startswith((" ", "\t")):
                break
            child_value = child.strip()
            if child_value.startswith("- "):
                items.append(parse_scalar(child_value[2:]))
            index += 1
        config[key] = items
    return config


def load_config(config_path: Path) -> dict[str, Any]:
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return simple_yaml_load(config_path)


def yaml_quote(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
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


def validate_required_config(config: dict[str, Any]) -> None:
    required_keys = ["algorithm_name", "network_name", "testcase", "episode", "index"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigError(f"Missing required YAML keys: {', '.join(missing_keys)}")


def resolve_seed_list(config: dict[str, Any]) -> list[int]:
    if "seeds" in config and config["seeds"] is not None:
        seeds_value = config["seeds"]
        if not isinstance(seeds_value, list):
            raise ConfigError("'seeds' must be a YAML list, for example: seeds: [42, 43, 44]")
        seeds = [int(seed) for seed in seeds_value]
    else:
        num_runs = int(config.get("num_runs", 10))
        seed_start = int(config.get("seed_start", 1))
        if num_runs <= 0:
            raise ConfigError("'num_runs' must be greater than zero")
        seeds = [seed_start + offset for offset in range(num_runs)]

    if not seeds:
        raise ConfigError("Seed list is empty")
    if len(seeds) != len(set(seeds)):
        raise ConfigError("Seed list contains duplicate values")
    return seeds


def resolve_max_workers(config: dict[str, Any], seed_count: int) -> int:
    configured = config.get("max_workers")
    if configured is None:
        cpu_count = os.cpu_count() or 1
        configured = max(1, min(seed_count, cpu_count // 2 or 1))
    max_workers = int(configured)
    if max_workers <= 0:
        raise ConfigError("'max_workers' must be greater than zero")
    if config.get("gui", False) and max_workers != 1:
        raise ConfigError("SUMO GUI runs are not safe for parallel execution. Use max_workers: 1.")
    return min(max_workers, seed_count)


def build_command(config: dict[str, Any], run_output_dir: Path, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--mode",
        "inference",
        "--testcase",
        str(config["testcase"]),
        "--episode",
        str(config["episode"]),
        "--index",
        str(config["index"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(run_output_dir),
    ]

    simulation_time = config.get("simulation_time")
    if simulation_time is None:
        simulation_time = config.get("steps")
    if simulation_time is not None:
        command.extend(["--steps", str(simulation_time)])

    if config.get("gui", False):
        command.append("--gui")
    if config.get("offline_infer", False):
        command.append("--offline_infer")
    if config.get("log", False):
        command.append("--log")

    return command


def copy_generated_data_file(testcase: str, run_output_dir: Path) -> None:
    source_file = PROJECT_ROOT / "network" / testcase / "data.xml"
    if source_file.exists():
        shutil.copy2(source_file, run_output_dir / "data.xml")


def get_next_run_number(base_output_dir: Path) -> int:
    existing_run_numbers = []
    for path in base_output_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"run_(\d+)", path.name)
        if match:
            existing_run_numbers.append(int(match.group(1)))
    return max(existing_run_numbers, default=0) + 1


def count_demand_vehicles(testcase: str, network_name: str) -> int | None:
    candidates = [
        PROJECT_ROOT / "network" / testcase / f"{testcase}.rou.xml",
        PROJECT_ROOT / "network" / network_name / f"{network_name}.rou.xml",
    ]
    route_file = next((path for path in candidates if path.exists()), None)
    if route_file is None:
        return None

    tree = ET.parse(route_file)
    root = tree.getroot()
    demand_ids = set()
    for tag_name in ("trip", "vehicle"):
        for element in root.findall(tag_name):
            element_id = element.get("id")
            if element_id is not None:
                demand_ids.add(element_id)
    return len(demand_ids) if demand_ids else None


def float_attr(element: ET.Element, name: str, default: float = 0.0) -> float:
    value = element.get(name)
    return default if value is None else float(value)


def int_attr(element: ET.Element, name: str, default: int = 0) -> int:
    value = element.get(name)
    return default if value is None else int(float(value))


def parse_tripinfo_metrics(tripinfo_path: Path) -> dict[str, Any]:
    if not tripinfo_path.exists():
        raise RunMetricsError(f"Missing tripinfo.xml: {tripinfo_path}")

    tree = ET.parse(tripinfo_path)
    trips = tree.getroot().findall("tripinfo")
    if not trips:
        raise RunMetricsError(f"No tripinfo records found in {tripinfo_path}")

    travel_times = [float_attr(trip, "duration") for trip in trips if trip.get("duration") is not None]
    waiting_times = [float_attr(trip, "waitingTime") for trip in trips if trip.get("waitingTime") is not None]
    arrivals = [float_attr(trip, "arrival") for trip in trips if trip.get("arrival") is not None]

    if not travel_times:
        raise RunMetricsError(f"No duration values found in {tripinfo_path}")

    return {
        "arrived_vehicles": len(trips),
        "mean_travel_time": statistics.mean(travel_times),
        "median_travel_time": statistics.median(travel_times),
        "min_travel_time": min(travel_times),
        "max_travel_time": max(travel_times),
        "std_travel_time": statistics.pstdev(travel_times) if len(travel_times) > 1 else 0.0,
        "mean_waiting_time": statistics.mean(waiting_times) if waiting_times else 0.0,
        "total_waiting_time": sum(waiting_times),
        "network_clear_time": max(arrivals) if arrivals else 0.0,
        "num_trips": len(trips),
    }


def parse_summary_metrics(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise RunMetricsError(f"Missing summary.xml: {summary_path}")

    tree = ET.parse(summary_path)
    steps = tree.getroot().findall("step")
    if not steps:
        raise RunMetricsError(f"No summary step records found in {summary_path}")

    final_step = steps[-1]
    return {
        "summary_final_time": float_attr(final_step, "time"),
        "summary_loaded": int_attr(final_step, "loaded"),
        "summary_inserted": int_attr(final_step, "inserted"),
        "summary_running": int_attr(final_step, "running"),
        "summary_waiting": int_attr(final_step, "waiting"),
        "summary_ended": int_attr(final_step, "ended"),
        "summary_arrived": int_attr(final_step, "arrived"),
        "summary_mean_waiting_time": float_attr(final_step, "meanWaitingTime", -1.0),
        "summary_mean_travel_time": float_attr(final_step, "meanTravelTime", -1.0),
        "summary_mean_speed": float_attr(final_step, "meanSpeed", -1.0),
    }


def command_to_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def prepend_env_path(env: dict[str, str], key: str, value: Path) -> None:
    existing = env.get(key, "")
    value_text = str(value)
    env[key] = value_text if not existing else value_text + os.pathsep + existing


def prepare_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    sumo_home = env.get("SUMO_HOME")
    if sumo_home:
        sumo_home_path = Path(sumo_home)
        prepend_env_path(env, "PATH", sumo_home_path / "bin")
        prepend_env_path(env, "PYTHONPATH", sumo_home_path / "tools")

    return env


def read_log_excerpt(log_file: str | Path, line_count: int = 40) -> str:
    path = Path(log_file)
    if not path.exists():
        return f"Log file does not exist: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    excerpt = lines[-line_count:]
    return "\n".join(excerpt)


def run_seed_job(job: SeedJob) -> dict[str, Any]:
    job.run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = job.run_output_dir / "run.log"
    start_time = time.time()
    env = prepare_worker_env()

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"seed={job.seed}\n")
        log_handle.write(f"command={command_to_text(job.command)}\n\n")
        log_handle.flush()
        process = subprocess.run(
            job.command,
            cwd=str(RUNTIME_DIR),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    duration_seconds = time.time() - start_time
    copy_generated_data_file(str(job.command[job.command.index("--testcase") + 1]), job.run_output_dir)

    result = {
        "run": job.run_output_dir.name,
        "run_number": job.run_number,
        "seed": job.seed,
        "run_dir": str(job.run_output_dir),
        "log_file": str(log_file),
        "command": command_to_text(job.command),
        "returncode": process.returncode,
        "duration_seconds": duration_seconds,
    }
    if process.returncode != 0:
        result["status"] = "failed"
        result["reason"] = f"subprocess returned {process.returncode}"
    else:
        result["status"] = "ran"
    return result


def summarize_run(run_result: dict[str, Any], demand_count: int | None) -> dict[str, Any]:
    record = dict(run_result)
    if record.get("status") == "failed":
        return record

    run_dir = Path(record["run_dir"])
    try:
        metrics = parse_tripinfo_metrics(run_dir / "tripinfo.xml")
        summary_metrics = parse_summary_metrics(run_dir / "summary.xml")
        record.update(metrics)
        record.update(summary_metrics)

        if abs(record["summary_arrived"] - record["arrived_vehicles"]) > SUMMARY_TRIPINFO_ARRIVAL_TOLERANCE:
            record["status"] = "incomplete"
            record["reason"] = (
                f"summary arrived {record['summary_arrived']} does not match "
                f"tripinfo count {record['arrived_vehicles']}"
            )
        elif demand_count is not None and record["arrived_vehicles"] < demand_count:
            record["status"] = "incomplete"
            record["reason"] = f"arrived {record['arrived_vehicles']} / demand {demand_count} vehicles"
        else:
            record["status"] = "complete"
            record["reason"] = ""

        record["demand_vehicles"] = demand_count if demand_count is not None else ""
        return record
    except Exception as error:
        record["status"] = "failed"
        record["reason"] = str(error)
        return record


def sort_records(records: list[dict[str, Any]], ranking_metric: str) -> list[dict[str, Any]]:
    valid_records = [
        record
        for record in records
        if record.get("status") == "complete"
        and ranking_metric in record
        and "arrived_vehicles" in record
        and "mean_waiting_time" in record
    ]
    return sorted(
        valid_records,
        key=lambda record: (
            float(record[ranking_metric]),
            -int(record["arrived_vehicles"]),
            float(record["mean_waiting_time"]),
            int(record["seed"]),
        ),
    )


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_records = [
        record
        for record in records
        if record.get("status") in {"complete", "incomplete"} and "mean_travel_time" in record
    ]
    aggregate: dict[str, Any] = {
        "total_runs": len(records),
        "parsed_runs": len(parsed_records),
        "complete_runs": sum(1 for record in records if record.get("status") == "complete"),
        "failed_runs": sum(1 for record in records if record.get("status") == "failed"),
        "incomplete_runs": sum(1 for record in records if record.get("status") == "incomplete"),
    }
    metric_names = [
        "mean_travel_time",
        "median_travel_time",
        "min_travel_time",
        "max_travel_time",
        "mean_waiting_time",
        "total_waiting_time",
        "network_clear_time",
        "arrived_vehicles",
    ]
    for metric_name in metric_names:
        values = [float(record[metric_name]) for record in parsed_records if metric_name in record]
        aggregate[metric_name] = statistics.mean(values) if values else 0.0
        aggregate[f"{metric_name}_std_across_runs"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return aggregate


def write_metrics_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_best_seeds(path: Path, sorted_records: list[dict[str, Any]], top_k: int, ranking_metric: str) -> None:
    best_seeds = []
    for rank, record in enumerate(sorted_records[:top_k], start=1):
        best_seeds.append(
            {
                "rank": rank,
                "seed": record["seed"],
                "run": record["run"],
                "run_dir": record["run_dir"],
                "status": record["status"],
                "mean_travel_time": round(float(record["mean_travel_time"]), 6),
                "mean_waiting_time": round(float(record["mean_waiting_time"]), 6),
                "total_waiting_time": round(float(record["total_waiting_time"]), 6),
                "arrived_vehicles": record["arrived_vehicles"],
                "demand_vehicles": record.get("demand_vehicles", ""),
                "summary_arrived": record["summary_arrived"],
                "network_clear_time": round(float(record["network_clear_time"]), 6),
                "command": record["command"],
            }
        )
    write_simple_yaml(
        path,
        {
            "ranking_metric": ranking_metric,
            "selection_top_k": top_k,
            "generated_at_unix": int(time.time()),
            "best_seeds": best_seeds,
        },
    )


def build_jobs(config: dict[str, Any], base_output_dir: Path, seeds: list[int]) -> list[SeedJob]:
    start_run_number = get_next_run_number(base_output_dir)
    jobs = []
    for offset, seed in enumerate(seeds):
        run_number = start_run_number + offset
        run_output_dir = base_output_dir / f"run_{run_number}"
        jobs.append(
            SeedJob(
                run_number=run_number,
                seed=seed,
                run_output_dir=run_output_dir,
                command=build_command(config, run_output_dir, seed),
            )
        )
    return jobs


def run_jobs(jobs: list[SeedJob], max_workers: int) -> tuple[list[dict[str, Any]], float]:
    start_time = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(run_seed_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "run": job.run_output_dir.name,
                    "run_number": job.run_number,
                    "seed": job.seed,
                    "run_dir": str(job.run_output_dir),
                    "log_file": str(job.run_output_dir / "run.log"),
                    "command": command_to_text(job.command),
                    "returncode": "",
                    "duration_seconds": 0.0,
                    "status": "failed",
                    "reason": str(error),
                }
            results.append(result)
            print(
                f"[{result['status']}] run={result['run']} seed={result['seed']} "
                f"duration={float(result['duration_seconds']):.2f}s"
            )
    return sorted(results, key=lambda result: int(result["run_number"])), time.time() - start_time


def run_pipeline(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_required_config(config)
    seeds = resolve_seed_list(config)
    max_workers = resolve_max_workers(config, len(seeds))
    ranking_metric = str(config.get("ranking_metric", "mean_travel_time"))
    top_k = int(config.get("selection_top_k", DEFAULT_TOP_K))
    explicit_seed_list = "seeds" in config and config["seeds"] is not None
    max_seed_windows = 1 if explicit_seed_list else int(config.get("max_seed_windows", 10))

    if ranking_metric != "mean_travel_time":
        raise ConfigError("Only ranking_metric: mean_travel_time is currently supported")
    if top_k <= 0:
        raise ConfigError("'selection_top_k' must be greater than zero")
    if max_seed_windows <= 0:
        raise ConfigError("'max_seed_windows' must be greater than zero")
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find main script at {MAIN_SCRIPT}")
    if not os.environ.get("SUMO_HOME"):
        raise EnvironmentError("SUMO_HOME is not set. Please set SUMO_HOME before running experiments.")

    base_output_dir = SCRIPT_DIR / str(config["algorithm_name"]) / str(config["network_name"])
    base_output_dir.mkdir(parents=True, exist_ok=True)

    demand_count = count_demand_vehicles(str(config["testcase"]), str(config["network_name"]))
    records: list[dict[str, Any]] = []
    total_wall_time_seconds = 0.0

    for window_index in range(max_seed_windows):
        if explicit_seed_list:
            window_seeds = seeds
        else:
            window_start = int(config.get("seed_start", 1)) + window_index * len(seeds)
            window_seeds = [window_start + offset for offset in range(len(seeds))]

        print(
            f"Running seed window {window_index + 1}/{max_seed_windows}: "
            f"{len(window_seeds)} seeds with max_workers={max_workers}"
        )
        jobs = build_jobs(config, base_output_dir, window_seeds)
        for job in jobs:
            print(f"queued run={job.run_output_dir.name} seed={job.seed}")

        run_results, wall_time_seconds = run_jobs(jobs, max_workers)
        total_wall_time_seconds += wall_time_seconds
        records.extend(summarize_run(result, demand_count) for result in run_results)

        complete_count = sum(1 for record in records if record.get("status") == "complete")
        if complete_count >= top_k:
            print(f"Candidate gate passed: {complete_count} complete seeds found.")
            break

        if explicit_seed_list:
            print(f"Candidate gate failed: only {complete_count}/{top_k} explicit seeds are complete.")
            break

        print(
            f"Candidate gate not met after window {window_index + 1}: "
            f"{complete_count}/{top_k} complete seeds found. Continuing with next seed window."
        )

    sorted_records = sort_records(records, ranking_metric)
    aggregate = aggregate_metrics(records)

    if records and aggregate["failed_runs"] == len(records):
        first_failed = records[0]
        print("\nAll seed runs failed. First failed run log excerpt:")
        print(read_log_excerpt(first_failed.get("log_file", "")))
        raise RuntimeError(
            "All seed runs failed before producing metrics. "
            f"First failed run: {first_failed.get('run')} seed={first_failed.get('seed')}. "
            f"See log: {first_failed.get('log_file')}"
        )

    csv_path = base_output_dir / "metrics_by_seed.csv"
    json_path = base_output_dir / "metrics_summary.json"
    best_seeds_path = base_output_dir / "best_seeds.yaml"

    write_metrics_csv(csv_path, records)
    write_best_seeds(best_seeds_path, sorted_records, top_k, ranking_metric)

    payload = {
        "config_path": str(config_path),
        "base_output_dir": str(base_output_dir),
        "max_workers": max_workers,
        "seed_count": len(records),
        "wall_time_seconds": total_wall_time_seconds,
        "ranking_metric": ranking_metric,
        "selection_top_k": top_k,
        "demand_vehicles": demand_count,
        "aggregate": aggregate,
        "runs": records,
        "best_seeds": sorted_records[:top_k],
    }
    with json_path.open("w", encoding="utf-8") as json_handle:
        json.dump(payload, json_handle, indent=2)

    print(f"Saved per-seed metrics to {csv_path}")
    print(f"Saved summary metrics to {json_path}")
    print(f"Saved best seeds to {best_seeds_path}")
    print(
        "Benchmark: "
        f"parallel_wall_time={total_wall_time_seconds:.2f}s, "
        f"max_workers={max_workers}, completed={aggregate['complete_runs']}/{len(records)}"
    )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel MARL inference seed search")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "experiment.yaml"),
        help="Path to the experiment YAML file",
    )
    args = parser.parse_args()
    run_pipeline(Path(args.config).resolve())


if __name__ == "__main__":
    main()
