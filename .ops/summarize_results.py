#!/usr/bin/env python3
"""Summarize Greyhound training and controller logs without interpreting outcomes.

Boolean evidence flags mean that matching log text was observed. They do not
establish that an experiment succeeded or that a performance claim is valid.
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


TRAINING_NAMES = ("console.log",)
GLOBAL_NAME_RE = re.compile(r"global_controller.*\.log$", re.IGNORECASE)
LOCAL_NAME_RE = re.compile(r"local_controller.*\.log$", re.IGNORECASE)
MEGATRON_NAME_RE = re.compile(r"megatron_output.*\.log$", re.IGNORECASE)
ITERATION_RE = re.compile(
    r"\biteration\s+(?P<index>\d+)\s*/\s*(?P<total>\d+)\b", re.IGNORECASE
)
ITERATION_TIME_RE = re.compile(
    r"(?:elapsed\s+time\s+per\s+iteration|iteration\s+time|iteration)"
    r"\s*\(\s*ms\s*\)\s*[:=]\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]?inf)",
    re.IGNORECASE,
)
CSV_METRIC_RE = re.compile(
    r"^\s*(?P<index>\d+)\s*,\s*(?P<value>"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]?inf)\s*$",
    re.IGNORECASE,
)
LOSS_RE = re.compile(
    r"\b(?P<label>lm\s+loss|loss)\s*[:=]\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]?inf)",
    re.IGNORECASE,
)
TIMESTAMP_RE = re.compile(
    r"(?<!\d)(?P<value>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)(?!\d)"
)
RANK_RE = re.compile(r"(?:global\s+)?rank\s*[=:]?\s*(\d+)", re.IGNORECASE)
FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|[+-]?inf", re.IGNORECASE)
COMPUTATION_RE = re.compile(
    r"Computation result of rank\s+(\d+).*?min\s*=\s*(%s).*?max\s*=\s*(%s)"
    r".*?avg\s*=\s*(%s).*?std\s*=\s*(%s)" % (FLOAT_RE.pattern, FLOAT_RE.pattern, FLOAT_RE.pattern, FLOAT_RE.pattern),
    re.IGNORECASE,
)
ACF_RE = re.compile(r"(?:estimated|esimated)\s+iteration\s+time\s*:\s*(\{.*\})", re.IGNORECASE)
DP_PLAN_RE = re.compile(r"new\s+DP\s+plan(?:\s+is|\s*:)\s*\[([^\]]*)\]", re.IGNORECASE)
DP_CHANGED_RE = re.compile(r"DP\s+changed\s*:\s*from\s*(\d+)\s*to\s*(\d+)", re.IGNORECASE)

EVIDENCE_LIMIT = 200
GENERIC_DIRS = {"results", "result", "logs", "log", "trainlog", "output", "outputs", "runs", "artifacts"}
RANK_SUFFIX_RE = re.compile(r"(?:[_-](?:node)?rank)[_-]?\d+$", re.IGNORECASE)


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def iso_timestamp(value: str) -> tuple[str | None, float | None]:
    value = value.strip().strip("[]()")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value.replace(",", "."))
    except ValueError:
        return None, None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z"), parsed.timestamp()


def extract_log_timestamp(line: str) -> tuple[str | None, float | None]:
    match = TIMESTAMP_RE.search(line)
    return iso_timestamp(match.group("value")) if match else (None, None)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def duration_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["time_ms"]) for row in records if row.get("time_ms") is not None and math.isfinite(float(row["time_ms"]))]
    return {
        "n": len(values),
        "median_ms": statistics.median(values) if values else None,
        "p90_ms": percentile(values, 0.90),
    }


def selected_experiment_key(path: Path, root: Path, injection_dirs: set[Path]) -> str:
    """Group files by an explicit injection directory or top-level run folder."""
    ancestors = [parent for parent in path.parents if parent == root or root in parent.parents]
    injection_ancestors = [parent for parent in ancestors if parent in injection_dirs]
    if injection_ancestors:
        return str(max(injection_ancestors, key=lambda item: len(item.parts)).relative_to(root) or Path("."))

    rel = path.relative_to(root)
    dirs = list(rel.parts[:-1])
    if not dirs:
        return "."
    run_component = next((part for part in dirs if part.lower() not in GENERIC_DIRS), dirs[0])
    run_component = RANK_SUFFIX_RE.sub("", run_component)
    return run_component or "."


def collect_paths(root: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    training: list[Path] = []
    global_logs: list[Path] = []
    local_logs: list[Path] = []
    injections: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if MEGATRON_NAME_RE.fullmatch(name) or name in TRAINING_NAMES:
            training.append(path)
        elif GLOBAL_NAME_RE.fullmatch(name):
            global_logs.append(path)
        elif LOCAL_NAME_RE.fullmatch(name):
            local_logs.append(path)
        elif name == "injection.jsonl":
            injections.append(path)
    return training, global_logs, local_logs, injections


def source_ref(path: Path, root: Path, line_no: int, text: str) -> dict[str, Any]:
    return {"file": str(path.relative_to(root)), "line": line_no, "text": text.strip()[:1000]}


def add_evidence(bucket: dict[str, list[dict[str, Any]]], key: str, item: dict[str, Any]) -> None:
    values = bucket.setdefault(key, [])
    if len(values) < EVIDENCE_LIMIT:
        values.append(item)


def loss_matches(line: str) -> dict[str, float | None]:
    found: dict[str, float | None] = {}
    for match in LOSS_RE.finditer(line):
        label = re.sub(r"\s+", " ", match.group("label").lower())
        found[label] = safe_float(match.group("value"))
    return found


def parse_training_file(path: Path, root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    csv_mode = False
    rank_match = re.search(r"megatron_output[_-](\d+)", path.name, re.IGNORECASE)
    file_rank = int(rank_match.group(1)) if rank_match else None
    for line_no, raw_line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
        line = raw_line.rstrip("\r\n")
        if "Iteration, IterationTime(ms)" in line:
            csv_mode = True
            continue

        index: int | None = None
        time_value: float | None = None
        iteration_match = ITERATION_RE.search(line)
        time_match = ITERATION_TIME_RE.search(line)
        if iteration_match and time_match:
            index = int(iteration_match.group("index"))
            time_value = safe_float(time_match.group("value"))
        elif csv_mode:
            csv_match = CSV_METRIC_RE.match(line)
            if csv_match:
                index = int(csv_match.group("index"))
                time_value = safe_float(csv_match.group("value"))
        if index is None:
            continue

        timestamp, timestamp_epoch = extract_log_timestamp(line)
        inline_rank_match = RANK_RE.search(line)
        rank = int(inline_rank_match.group(1)) if inline_rank_match else file_rank
        losses = loss_matches(line)
        finite_values = [math.isfinite(value) for value in losses.values() if value is not None]
        records.append({
            "index": index,
            "time_ms": time_value,
            "timestamp": timestamp,
            "timestamp_epoch": timestamp_epoch,
            "rank": rank,
            "losses": losses,
            "loss_finite": all(finite_values) if finite_values else None,
            "source": source_ref(path, root, line_no, line),
            "source_priority": 0 if MEGATRON_NAME_RE.fullmatch(path.name) else 1,
        })
    return records


def deduplicate_iterations(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_index[row["index"]].append(row)
    unique: list[dict[str, Any]] = []
    conflicts = 0
    for index in sorted(by_index):
        group = sorted(by_index[index], key=lambda item: (item["source_priority"], item["source"]["file"], item["source"]["line"]))
        canonical = group[0]
        timed = [row for row in group if row.get("time_ms") is not None]
        raw_timing_values = [float(row["time_ms"]) for row in timed]
        finite_timing_values = sorted({value for value in raw_timing_values if math.isfinite(value)})
        nonfinite_timing_values = sorted({"NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity") for value in raw_timing_values if not math.isfinite(value)})
        timing_values: list[Any] = finite_timing_values + nonfinite_timing_values
        distinct_timing_count = len(finite_timing_values) + len(nonfinite_timing_values)
        if distinct_timing_count > 1:
            conflicts += 1
        losses: dict[str, float | None] = {}
        for row in group:
            for label, value in row.get("losses", {}).items():
                losses.setdefault(label, value)
        timestamps = [row for row in group if row.get("timestamp_epoch") is not None]
        time_source = timed[0] if timed else canonical
        timestamp_source = timestamps[0] if timestamps else canonical
        finite_values = [math.isfinite(value) for value in losses.values() if value is not None]
        unique.append({
            "index": index,
            "time_ms": time_source.get("time_ms"),
            "losses": losses,
            "loss_finite": all(finite_values) if finite_values else None,
            "timestamp": timestamp_source.get("timestamp"),
            "rank": canonical.get("rank"),
            "source": canonical["source"],
            "duplicate_count": len(group) - 1,
            "timing_candidates_ms": timing_values,
            "duplicate_sources": [row["source"] for row in group[1:]],
            "timestamp_epoch": timestamp_source.get("timestamp_epoch"),
        })
    return unique, len(candidates) - len(unique), conflicts


def event_label(obj: dict[str, Any]) -> str:
    for key in ("scenario", "injection", "kind", "name", "event", "type", "action"):
        value = obj.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            label = str(value).strip()
            if key in {"event", "action"} and label.lower().replace("_", " ").replace("-", " ") in {
                "inject", "injection", "start", "begin", "restore", "stop", "end", "reset", "release", "on", "off",
            }:
                return "injection"
            return label
    return "injection event"


def event_kind(obj: dict[str, Any], label: str) -> str:
    words = " ".join(str(obj.get(key, "")) for key in ("phase", "status", "state", "action", "event", "type", "name", "injection", "kind", "scenario", "command", "cmd"))
    words = (words + " " + label).lower().replace("_", " ").replace("-", " ")
    if re.search(r"\b(end|stop|stopped|restore|restored|reset|resetting|disable|disabled|off|release)\b", words):
        return "end"
    if re.search(r"\b(start|begin|apply|applied|enable|enabled|on|inject|injected|lock)\b", words):
        return "start"
    return "point"


def numeric_event_time(value: Any, training_start_epoch: float | None, relative: bool = False) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, None
    if isinstance(value, str):
        iso, epoch = iso_timestamp(value)
        if epoch is not None:
            return epoch, iso
        try:
            value = float(value)
        except ValueError:
            return None, None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None, None
    value = float(value)
    if relative or abs(value) < 100_000_000:
        if training_start_epoch is None:
            return None, None
        epoch = training_start_epoch + value
    else:
        epoch = value / 1000.0 if abs(value) >= 100_000_000_000 else value
    return epoch, datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def parse_injection_files(paths: list[Path], root: Path, training_start_epoch: float | None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    absolute_time_keys = ("time_utc", "timestamp", "datetime", "at", "event_time", "time", "ts", "epoch", "epoch_s", "unix_time", "unix_timestamp", "time_unix_s", "ts_epoch")
    relative_time_keys = ("elapsed_s", "elapsed_sec", "elapsed_seconds", "offset_s", "seconds_since_start", "relative_time_s", "t_rel_s", "time_s")
    start_keys = ("start_time", "started_at", "start_ts", "start_epoch", "start_elapsed_s", "start")
    end_keys = ("end_time", "ended_at", "end_ts", "end_epoch", "end_elapsed_s", "end")
    for path in paths:
        for line_no, raw_line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            label = event_label(obj)
            base = {"label": label, "source": source_ref(path, root, line_no, line)}
            iteration = obj.get("iteration")
            if isinstance(iteration, int) and not isinstance(iteration, bool):
                base["iteration"] = iteration
            emitted = False
            for key in start_keys:
                if key in obj:
                    epoch, iso = numeric_event_time(obj[key], training_start_epoch, relative=key.endswith("elapsed_s"))
                    if epoch is not None:
                        events.append({**base, "kind": "start", "timestamp": iso, "timestamp_epoch": epoch})
                        emitted = True
                        break
            for key in end_keys:
                if key in obj:
                    epoch, iso = numeric_event_time(obj[key], training_start_epoch, relative=key.endswith("elapsed_s"))
                    if epoch is not None:
                        events.append({**base, "kind": "end", "timestamp": iso, "timestamp_epoch": epoch})
                        emitted = True
                        break
            if emitted:
                continue
            for key in absolute_time_keys:
                if key not in obj:
                    continue
                epoch, iso = numeric_event_time(obj[key], training_start_epoch)
                if epoch is not None:
                    events.append({**base, "kind": event_kind(obj, label), "timestamp": iso, "timestamp_epoch": epoch})
                    emitted = True
                    break
            if emitted:
                continue
            for key in relative_time_keys:
                if key not in obj:
                    continue
                epoch, iso = numeric_event_time(obj[key], training_start_epoch, relative=True)
                if epoch is not None:
                    events.append({**base, "kind": event_kind(obj, label), "timestamp": iso, "timestamp_epoch": epoch})
                    break
    events.sort(key=lambda item: (item["timestamp_epoch"], item["source"]["file"], item["source"]["line"]))
    return events


def event_intervals(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    open_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        label = event["label"]
        if event["kind"] == "start":
            open_events[label].append(event)
        elif event["kind"] == "end" and open_events.get(label):
            start = open_events[label].pop(0)
            if event["timestamp_epoch"] >= start["timestamp_epoch"]:
                intervals.append({
                    "label": label,
                    "start_epoch": start["timestamp_epoch"],
                    "end_epoch": event["timestamp_epoch"],
                    "start_timestamp": start["timestamp"],
                    "end_timestamp": event["timestamp"],
                })
    return sorted(intervals, key=lambda item: item["start_epoch"])


def phase_for_epoch(epoch: float, events: list[dict[str, Any]], intervals: list[dict[str, Any]]) -> str:
    for interval in intervals:
        if interval["start_epoch"] <= epoch <= interval["end_epoch"]:
            return f"during: {interval['label']}"
    if not events:
        return "unassigned"
    starts = [event for event in events if event["kind"] == "start"]
    if starts:
        first_start = min(starts, key=lambda event: event["timestamp_epoch"])
        if epoch < first_start["timestamp_epoch"]:
            return "baseline"
        prior_ends = [event for event in events if event["kind"] == "end" and event["timestamp_epoch"] <= epoch]
        if prior_ends:
            last_end = prior_ends[-1]
            later_starts = [event for event in starts if event["timestamp_epoch"] > last_end["timestamp_epoch"] and event["timestamp_epoch"] <= epoch]
            if later_starts:
                return f"after start: {later_starts[-1]['label']} (no matching end)"
            return "post-injection"
        prior_starts = [event for event in starts if event["timestamp_epoch"] <= epoch]
        if prior_starts:
            return f"after start: {prior_starts[-1]['label']} (no matching end)"
        return "baseline"
    prior_points = [event for event in events if event["timestamp_epoch"] <= epoch]
    if not prior_points:
        return "baseline"
    if len(prior_points) == len(events):
        return f"after: {prior_points[-1]['label']}"
    return f"between events: {prior_points[-1]['label']}"


def build_phase_stats(iterations: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"available": False, "reason": "injection.jsonl is absent or has no usable timestamps", "phases": []}
    intervals = event_intervals(events)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_timestamp = 0
    for row in iterations:
        if row.get("time_ms") is None or not math.isfinite(float(row["time_ms"])):
            continue
        epoch = row.get("timestamp_epoch")
        if epoch is None:
            missing_timestamp += 1
            continue
        groups[phase_for_epoch(epoch, events, intervals)].append(row)
    phases = []
    for name in sorted(groups):
        phases.append({"name": name, **duration_stats(groups[name]), "iteration_indices": [row["index"] for row in groups[name]]})
    return {
        "available": True,
        "method": "Training log timestamps are assigned to event-defined intervals; point events divide baseline and subsequent periods.",
        "event_intervals": intervals,
        "phases": phases,
        "iterations_without_timestamp": missing_timestamp,
    }


def new_experiment(key: str) -> dict[str, Any]:
    return {
        "experiment": key,
        "training_files": [],
        "global_controller_files": [],
        "local_controller_files": [],
        "injection_files": [],
        "iteration_candidates": [],
        "evidence": defaultdict(list),
        "computation_by_phase_rank": defaultdict(lambda: defaultdict(list)),
        "acf_estimates": [],
        "failslow_ranks": [],
        "root_reasons": [],
        "dp_plans": [],
        "dp_changes_applied": [],
    }


def parse_controller_file(path: Path, root: Path, exp: dict[str, Any], role: str) -> None:
    phase = "startup"
    for line_no, raw_line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
        line = raw_line.rstrip("\r\n")
        ref = source_ref(path, root, line_no, line)
        lower = line.lower()
        if "performing pre-check before training start" in lower or "performing precheck before training start" in lower:
            phase = "precheck"
            add_evidence(exp["evidence"], "precheck", ref)
        elif "performing validation" in lower:
            phase = "validation"
            add_evidence(exp["evidence"], "validation", ref)

        if re.search(r"computation tasks? (?:are )?dispatched", line, re.IGNORECASE):
            add_evidence(exp["evidence"], "computation_tasks_dispatched", ref)
        comp = COMPUTATION_RE.search(line)
        if comp:
            rank = comp.group(1)
            item = {"rank": int(rank), "min": safe_float(comp.group(2)), "max": safe_float(comp.group(3)), "avg": safe_float(comp.group(4)), "std": safe_float(comp.group(5)), "source": ref}
            exp["computation_by_phase_rank"][phase][rank].append(item)
            add_evidence(exp["evidence"], "computation_results", ref)

        acf = ACF_RE.search(line)
        if acf:
            ranks = {f"rank{rank}": safe_float(value) for rank, value in re.findall(r"['\"]?rank(\d+)['\"]?\s*:\s*(%s)" % FLOAT_RE.pattern, acf.group(1), re.IGNORECASE)}
            estimate = {"values_as_reported": ranks, "source": ref}
            exp["acf_estimates"].append(estimate)
            add_evidence(exp["evidence"], "acf_estimated_iteration_time", ref)

        if "failslow happens" in lower or "failslow events are reported" in lower or "failslow is reported" in lower:
            add_evidence(exp["evidence"], "failslow", ref)
            for rank in re.findall(r"rank\s*[=:]\s*(\d+)", line, re.IGNORECASE):
                if rank not in exp["failslow_ranks"]:
                    exp["failslow_ranks"].append(rank)
        if "profiling" in lower and ("reported" in lower or "mode" in lower or "profiling results" in lower):
            add_evidence(exp["evidence"], "profiling", ref)
        if "performing validation" in lower:
            add_evidence(exp["evidence"], "validation", ref)

        reason_match = re.search(r"(?:reason of fail[- ]slow is|root cause\s*:)\s*([^,;|]+)", line, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
            exp["root_reasons"].append({"value": reason, "source": ref})
            add_evidence(exp["evidence"], "root_reason", ref)
        plan_match = DP_PLAN_RE.search(line)
        if plan_match:
            values = [int(value) for value in re.findall(r"-?\d+", plan_match.group(1))]
            exp["dp_plans"].append({"values": values, "source": ref})
            add_evidence(exp["evidence"], "dp_plan_generated", ref)
        if re.search(r"\bAdjust\s+DP\b", line, re.IGNORECASE):
            add_evidence(exp["evidence"], "adjust_dp", ref)


def analyze_experiment(key: str, exp: dict[str, Any], root: Path) -> dict[str, Any]:
    unique, duplicate_count, conflict_count = deduplicate_iterations(exp["iteration_candidates"])
    for row in unique:
        row.pop("timestamp_epoch", None)
    # Keep private timestamps for event segmentation while keeping JSON rows portable.
    private_rows = deduplicate_iterations(exp["iteration_candidates"])[0]
    training_start_epoch = min((row["timestamp_epoch"] for row in private_rows if row.get("timestamp_epoch") is not None), default=None)
    injection_paths = [root / rel for rel in exp["injection_files"]]
    injection_events = parse_injection_files(injection_paths, root, training_start_epoch)
    phase_stats = build_phase_stats(private_rows, injection_events)

    losses = [row["loss_finite"] for row in unique if row.get("loss_finite") is not None]
    finite_count = sum(1 for value in losses if value)
    nonfinite_count = sum(1 for value in losses if not value)
    evidence_flags = {
        "precheck_observed": bool(exp["evidence"].get("precheck")),
        "computation_tasks_dispatched_observed": bool(exp["evidence"].get("computation_tasks_dispatched")),
        "per_rank_computation_results_observed": bool(exp["evidence"].get("computation_results")),
        "acf_iteration_estimates_observed": bool(exp["evidence"].get("acf_estimated_iteration_time")),
        "failslow_report_observed": bool(exp["evidence"].get("failslow")),
        "profiling_observed": bool(exp["evidence"].get("profiling")),
        "validation_observed": bool(exp["evidence"].get("validation")),
        "root_reason_observed": bool(exp["evidence"].get("root_reason")),
        "dp_plan_generated_observed": bool(exp["evidence"].get("dp_plan_generated")),
        "adjust_dp_observed": bool(exp["evidence"].get("adjust_dp")),
        "training_batch_distribution_applied_observed": bool(exp["evidence"].get("training_batch_distribution_applied")),
    }
    evidence = {
        "flags_semantics": "true means matching log evidence was found; it does not establish experiment success or a performance conclusion.",
        "flags": evidence_flags,
        "precheck": exp["evidence"].get("precheck", []),
        "computation_tasks_dispatched": exp["evidence"].get("computation_tasks_dispatched", []),
        "computation_by_phase_rank": {phase: dict(ranks) for phase, ranks in exp["computation_by_phase_rank"].items()},
        "acf_estimated_iteration_time": exp["acf_estimates"],
        "failslow": exp["evidence"].get("failslow", []),
        "failslow_ranks": sorted(exp["failslow_ranks"], key=lambda value: int(value)),
        "profiling": exp["evidence"].get("profiling", []),
        "validation": exp["evidence"].get("validation", []),
        "root_reasons": exp["root_reasons"],
        "dp_plans": exp["dp_plans"],
        "adjust_dp": exp["evidence"].get("adjust_dp", []),
        "training_batch_distribution_applied": exp["evidence"].get("training_batch_distribution_applied", []),
        "training_batch_distribution_changes": exp["dp_changes_applied"],
    }
    return {
        "experiment": key,
        "files": {
            "training": sorted(set(exp["training_files"])),
            "global_controller": sorted(set(exp["global_controller_files"])),
            "local_controller": sorted(set(exp["local_controller_files"])),
            "injection": sorted(set(exp["injection_files"])),
        },
        "iterations": unique,
        "iteration_summary": {
            "unique_iteration_count": len(unique),
            "duplicate_iteration_candidates_removed": duplicate_count,
            "iteration_timing_conflict_count": conflict_count,
            "first_iteration": min((row["index"] for row in unique), default=None),
            "last_iteration": max((row["index"] for row in unique), default=None),
            "loss_finite": {
                "observed": bool(losses),
                "all_observed_finite": all(losses) if losses else None,
                "finite_iteration_count": finite_count,
                "nonfinite_iteration_count": nonfinite_count,
            },
            "overall_timing": duration_stats(unique),
            "phase_timing": phase_stats,
        },
        "injection_events": injection_events,
        "evidence": evidence,
    }


def markdown_report(root: Path, experiments: list[dict[str, Any]]) -> str:
    lines = [
        "# Greyhound results summary",
        "",
        f"Results root: `{root}`",
        "",
        "> Evidence flags report matching log text only. They do not establish a performance result or experiment success.",
        "",
    ]
    for exp in experiments:
        stats = exp["iteration_summary"]
        loss = stats["loss_finite"]
        loss_text = "not observed" if not loss["observed"] else (
            "all observed values finite" if loss["all_observed_finite"] else "non-finite value observed"
        )
        timing = stats["overall_timing"]
        lines.extend([
            f"## {exp['experiment']}",
            "",
            f"Unique iterations: {stats['unique_iteration_count']} (range {stats['first_iteration']}–{stats['last_iteration']}); duplicate candidates removed: {stats['duplicate_iteration_candidates_removed']}.",
            f"Iteration timing: n={timing['n']}, median={format_number(timing['median_ms'])} ms, p90={format_number(timing['p90_ms'])} ms.",
            f"Loss finiteness: {loss_text} (finite iterations: {loss['finite_iteration_count']}, non-finite: {loss['nonfinite_iteration_count']}).",
            "",
            "Evidence flags (presence only):",
            "",
            "| Evidence | Observed |",
            "|---|---:|",
        ])
        for name, value in exp["evidence"]["flags"].items():
            lines.append(f"| `{name}` | {'yes' if value else 'no'} |")
        lines.append("")
        phase_stats = stats["phase_timing"]
        if phase_stats["available"]:
            lines.extend(["Phase timing from injection events:", "", "| Phase | n | Median (ms) | p90 (ms) |", "|---|---:|---:|---:|"])
            for phase in phase_stats["phases"]:
                lines.append(f"| {phase['name']} | {phase['n']} | {format_number(phase['median_ms'])} | {format_number(phase['p90_ms'])} |")
            if not phase_stats["phases"]:
                lines.append("| No timestamped iteration samples | 0 | n/a | n/a |")
            lines.append("")
        else:
            lines.extend([f"Phase timing unavailable: {phase_stats['reason']}.", ""])

        ranks = exp["evidence"]["computation_by_phase_rank"]
        if ranks:
            lines.extend(["Per-rank computation evidence:", ""])
            for phase, rank_map in ranks.items():
                lines.append(f"- {phase}: " + ", ".join(
                    f"rank {rank}: {len(items)} record(s), latest avg={format_number(items[-1].get('avg'))}"
                    for rank, items in sorted(rank_map.items(), key=lambda pair: int(pair[0]))
                ))
            lines.append("")
        if exp["evidence"]["root_reasons"]:
            lines.append("Root-reason text: " + "; ".join(item["value"] for item in exp["evidence"]["root_reasons"]))
            lines.append("")
        if exp["evidence"]["dp_plans"]:
            lines.append("DP plans logged: " + "; ".join(str(item["values"]) for item in exp["evidence"]["dp_plans"]))
            lines.append("")
        if exp["evidence"]["training_batch_distribution_changes"]:
            lines.append("Training log changes observed: " + "; ".join(
                f"microbatch count {item['from']} → {item['to']}"
                for item in exp["evidence"]["training_batch_distribution_changes"]
            ))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3f}" if math.isfinite(number) else str(number)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def summarize(root: Path) -> dict[str, Any]:
    training, global_logs, local_logs, injections = collect_paths(root)
    injection_dirs = {path.parent for path in injections}
    buckets: dict[str, dict[str, Any]] = {}

    def bucket_for(path: Path) -> dict[str, Any]:
        key = selected_experiment_key(path, root, injection_dirs)
        if key not in buckets:
            buckets[key] = new_experiment(key)
        return buckets[key]

    for path in training:
        exp = bucket_for(path)
        exp["training_files"].append(str(path.relative_to(root)))
        exp["iteration_candidates"].extend(parse_training_file(path, root))
    for path in global_logs:
        exp = bucket_for(path)
        exp["global_controller_files"].append(str(path.relative_to(root)))
        parse_controller_file(path, root, exp, "global")
    for path in local_logs:
        exp = bucket_for(path)
        exp["local_controller_files"].append(str(path.relative_to(root)))
        parse_controller_file(path, root, exp, "local")
    for path in injections:
        exp = bucket_for(path)
        exp["injection_files"].append(str(path.relative_to(root)))

    if not buckets:
        buckets["."] = new_experiment(".")

    # The Megatron consumer logs the concrete old/new microbatch count when it
    # observes a new Redis dp_version. This is stronger evidence than a solver
    # plan alone and is counted only in training output files.
    for key, exp in buckets.items():
        for path in training:
            if selected_experiment_key(path, root, injection_dirs) != key:
                continue
            for line_no, raw_line in enumerate(path.open("r", encoding="utf-8", errors="replace"), 1):
                line = raw_line.rstrip("\r\n")
                match = DP_CHANGED_RE.search(line)
                if match:
                    item = {"from": int(match.group(1)), "to": int(match.group(2)), "source": source_ref(path, root, line_no, line)}
                    exp["dp_changes_applied"].append(item)
                    add_evidence(exp["evidence"], "training_batch_distribution_applied", item["source"])

    experiments = [analyze_experiment(key, buckets[key], root) for key in sorted(buckets)]
    return json_safe({
        "schema_version": 1,
        "result_root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "flag_semantics": "Evidence flags indicate that matching text was found in logs; they are not performance conclusions.",
        "experiments": experiments,
    })


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="root directory containing one or more experiment logs")
    args = parser.parse_args(argv)
    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"results root is not a directory: {root}")
    summary = summarize(root)
    json_path = root / "summary.json"
    markdown_path = root / "summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(root, summary["experiments"]), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
