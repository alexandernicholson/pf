#!/usr/bin/env python3
"""Pool immutable benchmark repetitions and compare control/candidate medians.

Example: python3 bench/compare.py --control 042 045 --candidate 043 044 \
    --output bench/comparison-session3.json
Times are per invocation; each recorded repetition has equal weight. Output
hash disagreement is reported in the JSON and exits 2. Invalid, incomplete, or
incompatible input archives are rejected before writing a comparison.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def resolve_run(value: str, results: Path) -> Path:
    direct = Path(value).expanduser()
    if direct.exists():
        if direct.is_file() and direct.name in ("manifest.json", "records.json"):
            direct = direct.parent
        require(direct.is_dir(), f"not a run directory: {value}")
        return direct.resolve()
    matches = []
    for path in results.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^[^-]+-(\d+)(?:-|$)", path.name)
        if ((value.isdigit() and match and int(match[1]) == int(value))
                or path.name == value or path.name.split("-", 1)[-1] == value):
            matches.append(path.resolve())
    require(len(matches) == 1, f"run {value!r} resolved to {len(matches)} directories")
    return matches[0]


def load_run(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    records = read_json(path / "records.json")
    require(manifest.get("status") == "passed" and manifest.get("returncode") == 0
            and not manifest.get("timed_out") and not manifest.get("parse_errors"),
            f"run is not passed and complete: {path.name}")
    require(isinstance(records, list), f"records must be an array: {path.name}")
    completion = [r for r in records if r.get("type") == "completion"]
    require(len(completion) == 1 and completion[0].get("correctness_failures") == 0,
            f"missing successful completion: {path.name}")
    sources = manifest["sources"]
    for filename in ("pf.c", "kernel_bench.c", "run.py"):
        require(sources.get(filename, {}).get("sha256") == sha(path / "source" / filename),
                f"frozen source hash mismatch for {filename}: {path.name}")
    metadata = [r for r in records if r.get("type") == "metadata"]
    require(len(metadata) == 1, f"expected one metadata record: {path.name}")
    references, samples = {}, {}
    for record in records:
        kind = record.get("type")
        if kind not in ("correctness", "sample"):
            continue
        case = record["case"]
        if kind == "correctness":
            require(case not in references and record.get("failed_values") == 0
                    and record.get("checked_values", 0) > 0,
                    f"duplicate or failed reference for {case}: {path.name}")
            require(re.fullmatch(r"[0-9a-f]{16}", record.get("output_fnv1a64", "")) is not None,
                    f"invalid reference output hash for {case}: {path.name}")
            references[case] = record
        else:
            require(record.get("hash_stable") is True,
                    f"unstable sample for {case}: {path.name}")
            for key in ("ns_per_call", "loops", "process_cpu_ns"):
                value = record.get(key)
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value > 0,
                        f"invalid {key} for {case}: {path.name}")
            require(isinstance(record["loops"], int), f"noninteger loops: {path.name}/{case}")
            samples.setdefault(case, []).append(record)
    require(samples and samples.keys() == references.keys(),
            f"reference/sample case sets differ: {path.name}")
    reps = manifest["arguments"]["reps"]
    require(isinstance(reps, int) and reps > 0 and metadata[0].get("reps") == reps,
            f"invalid repetition count: {path.name}")
    for case, values in samples.items():
        require(len(values) == reps and sorted(v.get("rep", -1) for v in values) == list(range(reps)),
                f"incomplete/duplicate repetitions for {case}: {path.name}")
        require(all(v.get("output_fnv1a64") == references[case]["output_fnv1a64"] for v in values),
                f"sample/reference hash disagreement for {case}: {path.name}")
    host = read_json(path / "host-before.json")
    arguments = manifest["arguments"]
    match = re.match(r"^[^-]+-(\d+)(?:-|$)", path.name)
    provenance = {
        "id": match[1] if match else manifest.get("label", path.name),
        "path": str(path), "label": manifest.get("label"),
        "started_utc": manifest.get("started_utc"), "completed_utc": manifest.get("completed_utc"),
        "source_sha256": sources["pf.c"]["sha256"],
        "harness_sha256": sources["kernel_bench.c"]["sha256"],
        "runner_sha256": sources["run.py"]["sha256"],
        "manifest_sha256": sha(path / "manifest.json"), "records_sha256": sha(path / "records.json"),
        "compiler": {k: arguments.get(k) for k in ("cc", "cflags", "ldflags")},
        "compiler_version": manifest.get("compiler_version", {}).get("stdout"),
        "threads": {"parallel": arguments.get("parallel"),
                    "requested_pf_threads": host.get("environment", {}).get("PF_THREADS", "auto"),
                    "recorded_effective_threads": metadata[0].get("effective_threads"),
                    "affinity": host.get("affinity"),
                    "cpu_quota": host.get("raw_files", {}).get("/sys/fs/cgroup/cpu.max")},
        "metadata": metadata[0], "reps": reps, "target_ms": arguments.get("target_ms"),
    }
    return {"provenance": provenance, "references": references, "samples": samples}


def stats(values: list[float]) -> dict:
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def compare(control: list[dict], candidate: list[dict]) -> dict:
    runs = control + candidate
    cases = list(runs[0]["samples"])
    harnesses = {r["provenance"]["harness_sha256"] for r in runs}
    runners = {r["provenance"]["runner_sha256"] for r in runs}
    require(len(harnesses) == 1 and len(runners) == 1,
            "harness/runner hashes differ; rerun all controls and candidates with the same harness")
    require(all(set(r["samples"]) == set(cases) for r in runs),
            "case sets differ; choose runs with identical workload coverage")
    output = {}
    for case in cases:
        item = {}
        for metric in ("wall_ns_per_call", "cpu_ns_per_call"):
            groups = {}
            for name, selected in (("control", control), ("candidate", candidate)):
                values = [s["ns_per_call"] if metric.startswith("wall")
                          else s["process_cpu_ns"] / s["loops"]
                          for run in selected for s in run["samples"][case]]
                groups[name] = stats(values)
            groups["speedup"] = groups["control"]["median"] / groups["candidate"]["median"]
            item[metric] = groups
        hashes = {name: sorted({run["references"][case]["output_fnv1a64"] for run in selected})
                  for name, selected in (("control", control), ("candidate", candidate))}
        item["correctness"] = {"output_hashes": hashes,
                               "exact_hash_match": len(set(hashes["control"] + hashes["candidate"])) == 1,
                               "failed_reference_values": 0, "unstable_samples": 0}
        output[case] = item
    return {
        "schema_version": 1, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "Pool raw repetitions with equal weight; speedup = control median / candidate median.",
        "cpu_definition": "process_cpu_ns / loops: aggregate process CPU time per invocation, in nanoseconds.",
        "validation": {"all_runs_passed_complete": True, "identical_case_sets": True,
                       "identical_harness_and_runner": True, "failed_reference_values": 0,
                       "unstable_samples": 0,
                       "all_case_hashes_match": all(v["correctness"]["exact_hash_match"] for v in output.values())},
        "control": [r["provenance"] for r in control],
        "candidate": [r["provenance"] for r in candidate], "cases": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", required=True, help="Run IDs or archive paths")
    parser.add_argument("--candidate", nargs="+", required=True, help="Run IDs or archive paths")
    parser.add_argument("--results", type=Path, default=ROOT / "bench/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        control_paths = [resolve_run(v, args.results) for v in args.control]
        candidate_paths = [resolve_run(v, args.results) for v in args.candidate]
        paths = control_paths + candidate_paths
        require(len(set(paths)) == len(paths), "duplicate or overlapping control/candidate runs")
        result = compare([load_run(p) for p in control_paths], [load_run(p) for p in candidate_paths])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    for case, values in result["cases"].items():
        wall, cpu = values["wall_ns_per_call"], values["cpu_ns_per_call"]
        print(f"{case:30s} {wall['control']['median']/1e6:9.4f} -> "
              f"{wall['candidate']['median']/1e6:9.4f} ms  "
              f"wall {wall['speedup']:.3f}x  CPU {cpu['speedup']:.3f}x")
    identical = result["validation"]["all_case_hashes_match"]
    print(f"{'Exact hashes match' if identical else 'HASH MISMATCH'}: {args.output}")
    return 0 if identical else 2


if __name__ == "__main__":
    sys.exit(main())
