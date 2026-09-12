#!/usr/bin/env python3
"""Compile frozen sources, run one serial benchmark process, retain raw evidence.

No dependencies beyond Python 3 and a C compiler. Timing processes are protected
by a shared advisory lock, including when comparing separate source snapshots.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except (OSError, PermissionError):
        return None


def command(args: list[str], cwd: Path = ROOT) -> dict:
    begin = time.monotonic_ns()
    try:
        p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)
        return dict(argv=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
                    wall_ns=time.monotonic_ns()-begin)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(argv=args, error=str(exc), wall_ns=time.monotonic_ns()-begin)


def machine_state() -> dict:
    paths = ["/proc/cpuinfo", "/proc/meminfo", "/proc/loadavg", "/proc/self/status",
             "/proc/self/cgroup", "/sys/devices/system/clocksource/clocksource0/current_clocksource",
             "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu.pressure",
             "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/memory.max",
             "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.events"]
    return {"captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "uname": list(platform.uname()), "python": sys.version,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "raw_files": {p: read(Path(p)) for p in paths},
            "lscpu": command(["lscpu", "--json"]),
            "environment": {k: v for k, v in os.environ.items()
                            if k.startswith(("PF_", "OMP_", "MKL_", "OPENBLAS_"))
                            or k in ("CC", "CFLAGS", "LDFLAGS")}}


def write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    a = sorted(values)
    idx = (len(a)-1)*q
    low = int(idx)
    return a[low] + (a[min(low+1, len(a)-1)]-a[low])*(idx-low)


def summarize(records: list[dict]) -> dict:
    names = list(dict.fromkeys(r["case"] for r in records if r.get("type") == "sample"))
    out = {}
    for name in names:
        samples = [r for r in records if r.get("type") == "sample" and r["case"] == name]
        values = [r["ns_per_call"] for r in samples]
        out[name] = {"n": len(values), "median_ns": statistics.median(values),
                     "min_ns": min(values), "max_ns": max(values),
                     "mean_ns": statistics.mean(values), "p10_ns": percentile(values, .1),
                     "p90_ns": percentile(values, .9),
                     "stdev_ns": statistics.stdev(values) if len(values) > 1 else 0,
                     "hashes_stable": all(r.get("hash_stable", False) for r in samples)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--source", type=Path, default=ROOT / "pf.c")
    p.add_argument("--cc", default=os.environ.get("CC", "cc"))
    p.add_argument("--cflags", default="-D_GNU_SOURCE -O3 -std=c11 -Wall -Wextra -march=native")
    p.add_argument("--ldflags", default="-lm -pthread")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--target-ms", type=float, default=25)
    p.add_argument("--profile", choices=["quick", "extended", "diverse"], default="quick")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--results", type=Path, default=ROOT / "bench/results")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--note", default="")
    a = p.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", a.label).strip("-.") or "run"
    dest = a.results.resolve() / f"{stamp}-{label}"
    dest.mkdir(parents=True, exist_ok=False)
    frozen = dest / "source"
    frozen.mkdir()
    # Freeze the exact translation unit and generated header before building.
    sources = {"pf.c": a.source.resolve(), "unicode_tables.h": ROOT / "unicode_tables.h",
               "kernel_bench.c": ROOT / "bench/kernel_bench.c", "run.py": Path(__file__).resolve()}
    if (ROOT / "Makefile").exists():
        sources["Makefile"] = ROOT / "Makefile"
    for name, src in sources.items():
        shutil.copyfile(src, frozen / name)
    (dest / "source.patch").write_text(command([
        "git", "diff", "--binary", "HEAD", "--", "pf.c", "Makefile", "bench", ":(exclude)bench/results"
    ]).get("stdout", ""))
    manifest = {"schema_version": 1, "label": a.label, "note": a.note,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "building", "synthetic_weights": True,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "git_head": command(["git", "rev-parse", "HEAD"]),
                "git_status": command(["git", "status", "--porcelain=v1"]),
                "sources": {name: {"original_path": str(src), "sha256": sha(frozen / name)}
                            for name, src in sources.items()},
                "compiler_version": command(shlex.split(a.cc) + ["--version"])}
    write_json(dest / "manifest.json", manifest)
    binary = dest / "kernel_bench"
    compile_args = shlex.split(a.cc) + shlex.split(a.cflags) + [
        '-DPF_BENCH_SOURCE="' + str(frozen / "pf.c") + '"',
        "-I", str(frozen), str(frozen / "kernel_bench.c"), "-o", str(binary)
    ] + shlex.split(a.ldflags)
    build = command(compile_args)
    (dest / "compile.stdout").write_text(build.pop("stdout", ""))
    (dest / "compile.stderr").write_text(build.pop("stderr", ""))
    manifest["build"] = build
    if build.get("returncode") != 0:
        manifest["status"] = "compile_failed"
        write_json(dest / "manifest.json", manifest)
        print(f"compile failed: {dest}", file=sys.stderr)
        print((dest / "compile.stderr").read_text(), file=sys.stderr)
        return 1
    manifest["binary_sha256"] = sha(binary)
    run_args = [str(binary), "--reps", str(a.reps), "--target-ms", str(a.target_ms),
                "--profile", a.profile] + (["--parallel"] if a.parallel else [])
    # Shared across repositories/snapshots in this machine; never time two suite
    # processes concurrently. Compiler and provenance gathering are not timed.
    with Path("/tmp/pf-cloud-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(dest / "host-before.json", machine_state())
        manifest["status"] = "running"
        manifest["run_argv"] = run_args
        write_json(dest / "manifest.json", manifest)
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic_ns()
        with (dest / "stdout.jsonl").open("w") as stdout, (dest / "stderr.txt").open("w") as stderr:
            proc = subprocess.Popen(run_args, cwd=ROOT, stdout=stdout, stderr=stderr)
            try:
                code = proc.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
                manifest["timed_out"] = True
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        manifest["process_wall_ns"] = time.monotonic_ns()-start
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        manifest["process_rusage"] = {
            name: getattr(usage_after, name)-getattr(usage_before, name)
            for name in ("ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")}
        # RUSAGE_CHILDREN high water is cumulative and can include the compiler;
        # individual C sample max_rss_kib is the benchmark process RSS evidence.
        manifest["cumulative_children_maxrss_kib"] = usage_after.ru_maxrss
        write_json(dest / "host-after.json", machine_state())
    records, parse_errors = [], []
    for lineno, line in enumerate((dest / "stdout.jsonl").read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            parse_errors.append({"line": lineno, "text": line, "error": str(exc)})
    # Retain every raw record/key, including future metrics unknown to this runner.
    write_json(dest / "records.json", records)
    summary = summarize(records)
    completed = any(r.get("type") == "completion" and r.get("correctness_failures") == 0 for r in records)
    valid = code == 0 and completed and not parse_errors and all(v["hashes_stable"] for v in summary.values())
    manifest.update(status="passed" if valid else "failed", returncode=code, parse_errors=parse_errors,
                    completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), summary=summary)
    write_json(dest / "manifest.json", manifest)
    write_json(dest / "summary.json", summary)
    for name, stats in summary.items():
        print(f"{name:30s} {stats['median_ns']/1e6:10.4f} ms  "
              f"range {stats['min_ns']/1e6:.4f}..{stats['max_ns']/1e6:.4f}")
    print(f"{manifest['status']}: {dest}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
