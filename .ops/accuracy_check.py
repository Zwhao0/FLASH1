#!/usr/bin/env python3
"""Compare Greyhound local-controller ACF estimates with recent Megatron timings.

The script only reads experiment logs. It writes accuracy-check.json and
accuracy-check.md under the supplied results root (default: repository/results).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRAINING_FILE_RE = re.compile(r"(?:megatron_output.*\.log|console\.log)$", re.IGNORECASE)
LOCAL_FILE_RE = re.compile(r"local_controller.*\.log$", re.IGNORECASE)
ITERATION_RE = re.compile(
    r"\biteration\s+(?P<index>\d+)\s*/\s*(?P<total>\d+)\b.*?"
    r"elapsed\s+time\s+per\s+iteration\s*\(\s*ms\s*\)\s*:\s*"
    r"(?P<ms>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]?inf)",
    re.IGNORECASE,
)
ACF_RE = re.compile(r"(?:estimated|esimated)\s+iteration\s+time\s*:\s*(\{[^}]*\})", re.IGNORECASE)
RANK_VALUE_RE = re.compile(
    r"['\"]?rank(?P<rank>\d+)['\"]?\s*:\s*(?P<value>"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]?inf)",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)"
)

# Ignore only intervals at least four times the run median and at least 2 s.
# This removes startup/pre-check pauses while retaining ordinary iteration jitter.
OUTLIER_RATIO = 4.0
OUTLIER_FLOOR_MS = 2000.0


def parse_timestamp(value: str) -> tuple[str | None, float | None]:
    value = value.strip().replace(",", ".")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None, None
    if dt.tzinfo is None:
        # The training and controller logs use the same host clock; UTC here is
        # only a common ordering frame for naive timestamps.
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z"), dt.timestamp()


def timestamp_in_line(line: str) -> tuple[str | None, float | None]:
    match = TIMESTAMP_RE.search(line)
    return parse_timestamp(match.group("timestamp")) if match else (None, None)


def finite_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_ape_percent": statistics.median(values) if values else None,
        "p90_ape_percent": percentile(values, 0.90),
    }


def load_training(paths: list[Path], results_root: Path) -> tuple[list[dict[str, Any]], int]:
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        priority = 0 if path.name.lower().startswith("megatron_output") else 1
        for line_no, raw in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
            match = ITERATION_RE.search(raw)
            if not match:
                continue
            timestamp, epoch = timestamp_in_line(raw)
            ms = finite_float(match.group("ms"))
            candidates[int(match.group("index"))].append({
                "index": int(match.group("index")),
                "time_ms": ms,
                "timestamp": timestamp,
                "timestamp_epoch": epoch,
                "source_priority": priority,
                "source": {"file": str(path.relative_to(results_root)), "line": line_no},
            })

    iterations = []
    duplicates = 0
    for index, group in sorted(candidates.items()):
        group.sort(key=lambda row: (row["source_priority"], row["source"]["file"], row["source"]["line"]))
        # console.log commonly mirrors megatron_output*.log; keep one sample per index.
        chosen = next((row for row in group if row["time_ms"] is not None), group[0])
        iterations.append({key: value for key, value in chosen.items() if key != "source_priority"})
        duplicates += len(group) - 1
    return iterations, duplicates


def load_acf(paths: list[Path], results_root: Path) -> list[dict[str, Any]]:
    observations = []
    for path in paths:
        lines = path.open("r", encoding="utf-8", errors="replace")
        for line_no, raw in enumerate(lines, 1):
            match = ACF_RE.search(raw)
            if not match:
                continue
            timestamp, epoch = timestamp_in_line(raw)
            ranks: dict[str, float] = {}
            for rank_match in RANK_VALUE_RE.finditer(match.group(1)):
                value = finite_float(rank_match.group("value"))
                if value is not None:
                    ranks[f"rank{int(rank_match.group('rank'))}"] = value
            observations.append({
                "timestamp": timestamp,
                "timestamp_epoch": epoch,
                "estimates_us": ranks,
                "source": {"file": str(path.relative_to(results_root)), "line": line_no},
            })
    observations.sort(key=lambda row: (row["timestamp_epoch"] if row["timestamp_epoch"] is not None else math.inf,
                                       row["source"]["file"], row["source"]["line"]))
    return observations


def load_injection_events(paths: list[Path], results_root: Path) -> list[dict[str, Any]]:
    events = []
    for path in paths:
        for line_no, raw in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            timestamp_value = obj.get("time_utc") or obj.get("timestamp") or obj.get("time")
            if not isinstance(timestamp_value, str):
                continue
            timestamp, epoch = parse_timestamp(timestamp_value)
            if epoch is None:
                continue
            event = str(obj.get("event", obj.get("action", ""))).strip().lower().replace("-", "_")
            events.append({
                "event": event,
                "timestamp": timestamp,
                "timestamp_epoch": epoch,
                "iteration": obj.get("iteration") if isinstance(obj.get("iteration"), int) else None,
                "graphics_mhz": obj.get("graphics_mhz"),
                "command": obj.get("command"),
                "source": {"file": str(path.relative_to(results_root)), "line": line_no},
            })
    return sorted(events, key=lambda event: (event["timestamp_epoch"], event["source"]["file"], event["source"]["line"]))


def phase_at(epoch: float | None, events: list[dict[str, Any]]) -> str:
    if epoch is None or not events:
        return "unavailable"
    state = "baseline"
    for event in events:
        if event["timestamp_epoch"] > epoch:
            break
        if event["event"] in {"inject", "injection", "start", "apply"}:
            state = "injection"
        elif event["event"] in {"restore", "restored", "reset", "stop", "release"}:
            state = "recovered"
    return state


def analyze_run(run_name: str, paths: list[Path], results_root: Path) -> dict[str, Any]:
    training = [p for p in paths if TRAINING_FILE_RE.fullmatch(p.name)]
    local = [p for p in paths if LOCAL_FILE_RE.fullmatch(p.name)]
    injections = [p for p in paths if p.name == "injection.jsonl"]
    iterations, duplicate_count = load_training(training, results_root)
    observations = load_acf(local, results_root)
    events = load_injection_events(injections, results_root)

    valid_times = [float(row["time_ms"]) for row in iterations
                   if row["time_ms"] is not None and math.isfinite(float(row["time_ms"])) and float(row["time_ms"]) > 0]
    run_median = statistics.median(valid_times) if valid_times else None
    outlier_threshold = max(OUTLIER_RATIO * run_median, OUTLIER_FLOOR_MS) if run_median is not None else None
    excluded = []
    eligible = []
    for row in iterations:
        ms = row["time_ms"]
        if ms is None or not math.isfinite(float(ms)) or float(ms) <= 0:
            excluded.append({"iteration": row["index"], "reason": "missing, non-finite, or non-positive timing", "time_ms": ms})
        elif outlier_threshold is not None and float(ms) > outlier_threshold:
            excluded.append({"iteration": row["index"], "reason": "long interval above outlier threshold", "time_ms": ms})
        elif row["timestamp_epoch"] is None:
            excluded.append({"iteration": row["index"], "reason": "no timestamp for time alignment", "time_ms": ms})
        else:
            eligible.append(row)
    eligible.sort(key=lambda row: (row["timestamp_epoch"], row["index"]))

    rank_errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    estimate_rows = []
    unaligned_observations = 0
    for observation in observations:
        epoch = observation["timestamp_epoch"]
        preceding = [row for row in eligible if epoch is not None and row["timestamp_epoch"] < epoch]
        reference = preceding[-2:]
        enough_reference = len(reference) == 2
        ref_ms = statistics.median([float(row["time_ms"]) for row in reference]) if enough_reference else None
        phase = phase_at(epoch, events)
        if epoch is None:
            unaligned_observations += 1
        rank_results = {}
        for rank, estimate_us in observation["estimates_us"].items():
            estimate_ms = estimate_us / 1000.0
            ape = (abs(estimate_ms - ref_ms) / ref_ms * 100.0) if ref_ms and enough_reference else None
            detail = {
                "estimate_us": estimate_us,
                "estimate_ms": estimate_ms,
                "reference_median_ms": ref_ms,
                "reference_iterations": [row["index"] for row in reference],
                "reference_iteration_times_ms": [row["time_ms"] for row in reference],
                "ape_percent": ape,
            }
            rank_results[rank] = detail
            if ape is not None and math.isfinite(ape):
                rank_errors[rank].append({"ape_percent": ape, "phase": phase, "timestamp": observation["timestamp"]})
        estimate_rows.append({
            "timestamp": observation["timestamp"],
            "phase": phase,
            "reference_available": enough_reference,
            "reference_iterations": [row["index"] for row in reference],
            "ranks": rank_results,
            "source": observation["source"],
        })

    all_errors = [item["ape_percent"] for values in rank_errors.values() for item in values]
    phase_names = ["baseline", "injection", "recovered", "unavailable"]
    by_phase = {}
    for phase in phase_names:
        phase_errors = [item["ape_percent"] for values in rank_errors.values() for item in values if item["phase"] == phase]
        by_phase[phase] = stats(phase_errors)
    per_rank = {}
    for rank in sorted(rank_errors, key=lambda name: int(name.replace("rank", ""))):
        per_rank[rank] = {
            **stats([item["ape_percent"] for item in rank_errors[rank]]),
            "by_phase": {
                phase: stats([item["ape_percent"] for item in rank_errors[rank] if item["phase"] == phase])
                for phase in phase_names
            },
        }

    return {
        "run": run_name,
        "inputs": {
            "training_logs": sorted(str(path.relative_to(results_root)) for path in training),
            "local_controller_logs": sorted(str(path.relative_to(results_root)) for path in local),
            "injection_files": sorted(str(path.relative_to(results_root)) for path in injections),
        },
        "iteration_alignment": {
            "iteration_count": len(iterations),
            "deduplicated_log_records": duplicate_count,
            "valid_reference_iteration_count": len(eligible),
            "excluded_iterations": excluded,
            "outlier_rule": f"exclude elapsed_ms > max({OUTLIER_RATIO:g} * run_median_ms, {OUTLIER_FLOOR_MS:g} ms); also exclude invalid timing or untimestamped rows",
            "run_median_ms_before_filter": run_median,
            "long_interval_threshold_ms": outlier_threshold,
            "reference_rule": "For each ACF timestamp, use the two latest eligible Megatron iterations with timestamps strictly earlier than the estimate; ground truth is their median.",
        },
        "injection_events": events,
        "phase_availability": {
            "available": bool(events),
            "reason": None if events else "No timestamped injection.jsonl events were found; only overall accuracy is meaningful.",
        },
        "acf_observations": estimate_rows,
        "accuracy": {
            "units": {"acf_log": "microseconds", "megaton_iteration_log": "milliseconds", "error": "absolute percentage error"},
            "overall": stats(all_errors),
            "by_rank": per_rank,
            "by_phase": by_phase,
            "observations_without_timestamp": unaligned_observations,
        },
        "limitations": [
            "Megatron elapsed time is treated as ground truth; this comparison does not independently validate whether that field captures all distributed-rank bottlenecks.",
            "ACF estimates are compared with the preceding two eligible logged iterations, not a future iteration; estimates can repeat between log updates.",
            "Long intervals are removed using the stated duration threshold; the logs do not label every pause as setup or pre-check, so exclusions are objective outlier filtering rather than proof of cause.",
            "Phase labels use the ACF observation timestamp. An estimate close to an injection boundary may reference iterations from the preceding phase.",
        ],
    }


def discover_runs(results_root: Path) -> dict[str, list[Path]]:
    candidates = []
    for path in sorted(results_root.rglob("*")):
        if path.is_file() and (TRAINING_FILE_RE.fullmatch(path.name) or LOCAL_FILE_RE.fullmatch(path.name) or path.name == "injection.jsonl"):
            candidates.append(path)
    runs: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        rel = path.relative_to(results_root)
        # Results are laid out as <results>/<run>/...; grouping by this first
        # component also accommodates a future results/dp-failslow/ tree.
        run_name = rel.parts[0] if len(rel.parts) > 1 else "."
        runs[run_name].append(path)
    return runs


def md_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def markdown_report(root: Path, runs: list[dict[str, Any]]) -> str:
    lines = [
        "# ACF accuracy check",
        "",
        f"Results root: `{root}`",
        "",
        "APE = |ACF estimate − reference| / reference × 100%. ACF µs values are converted to ms.",
        "Reference is the median of the two latest eligible Megatron iterations strictly before each ACF timestamp.",
        "Long intervals are excluded when elapsed time exceeds `max(4 × run median, 2000 ms)`; this is outlier filtering, not proof of pause cause.",
        "",
    ]
    for run in runs:
        align = run["iteration_alignment"]
        accuracy = run["accuracy"]
        overall = accuracy["overall"]
        lines.extend([
            f"## {run['run']}",
            "",
            f"Iterations parsed: {align['iteration_count']}; duplicate records removed: {align['deduplicated_log_records']}; eligible references: {align['valid_reference_iteration_count']}.",
            f"Excluded iterations: {', '.join(str(item['iteration']) for item in align['excluded_iterations']) or 'none'}; threshold: {md_number(align['long_interval_threshold_ms'])} ms.",
            f"Overall APE: n={overall['n']}, median={md_number(overall['median_ape_percent'])}%, p90={md_number(overall['p90_ape_percent'])}%.",
            f"Phase split: {'available' if run['phase_availability']['available'] else 'unavailable'}" + (f" ({run['phase_availability']['reason']})" if not run['phase_availability']['available'] else ""),
            "",
            "| Rank | Samples | Median APE | p90 APE |",
            "|---|---:|---:|---:|",
        ])
        if accuracy["by_rank"]:
            for rank, item in accuracy["by_rank"].items():
                lines.append(f"| {rank} | {item['n']} | {md_number(item['median_ape_percent'])}% | {md_number(item['p90_ape_percent'])}% |")
        else:
            lines.append("| no matched estimates | 0 | n/a | n/a |")
        lines.extend(["", "| Phase | Samples | Median APE | p90 APE |", "|---|---:|---:|---:|"])
        for phase, item in accuracy["by_phase"].items():
            if item["n"]:
                lines.append(f"| {phase} | {item['n']} | {md_number(item['median_ape_percent'])}% | {md_number(item['p90_ape_percent'])}% |")
        lines.extend(["", "### Limitations", ""])
        for limitation in run["limitations"]:
            lines.append(f"- {limitation}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main(argv: Iterable[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[1] / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", nargs="?", type=Path, default=default_root,
                        help="directory with run folders; defaults to repository/results")
    args = parser.parse_args(argv)
    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"results root is not a directory: {root}")
    run_inputs = discover_runs(root)
    runs = [analyze_run(name, paths, root) for name, paths in sorted(run_inputs.items())]
    report = {
        "schema_version": 1,
        "results_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "Independent comparison of controller ACF estimates to the median of the two nearest prior eligible Megatron iteration timings.",
        "runs": runs,
    }
    json_path = root / "accuracy-check.json"
    md_path = root / "accuracy-check.md"
    json_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(root, runs), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
