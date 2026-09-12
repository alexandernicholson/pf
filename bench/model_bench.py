#!/usr/bin/env python3
"""Immutable model regression and CLI benchmark records; Python standard library only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def counters():
    return {n: (Path("/sys/fs/cgroup") / n).read_text() for n in
            ("cpu.stat", "memory.current", "memory.events") if (Path("/sys/fs/cgroup") / n).is_file()}


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def run(out, name, cmd, env, timeout):
    record = {"name": name, "argv": list(map(str, cmd)), "cwd": str(Path.cwd()),
              "env_overrides": env, "timeout_s": timeout, "cgroup_before": counters(),
              "started_utc": time.strftime("%FT%TZ", time.gmtime())}
    save(out / (name + ".command.json"), record)
    start = time.monotonic()
    with (out / (name + ".stdout")).open("wb") as stdout, (out / (name + ".stderr")).open("wb") as stderr:
        try:
            p = subprocess.Popen(record["argv"], stdout=stdout, stderr=stderr,
                                 env={**os.environ, **env}, start_new_session=True)
            while True:
                pid, status, usage = os.wait4(p.pid, os.WNOHANG)
                if pid:
                    p.returncode = os.waitstatus_to_exitcode(status)
                    record["exit_code"] = p.returncode
                    record["rusage"] = {key[3:]: getattr(usage, key) for key in dir(usage) if key.startswith("ru_")}
                    break
                if time.monotonic() - start >= timeout:
                    record["timed_out"] = True
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    _, status, usage = os.wait4(p.pid, 0)
                    p.returncode = os.waitstatus_to_exitcode(status)
                    record.update(exit_code=p.returncode, rusage={key[3:]: getattr(usage, key) for key in dir(usage) if key.startswith("ru_")})
                    break
                time.sleep(0.01)
        except OSError as exc:
            record.update(exit_code=None, error=str(exc))
    record["wall_s"] = time.monotonic() - start
    record["cgroup_after"] = counters()
    for stream in ("stdout", "stderr"):
        record[stream] = identity(out / (name + "." + stream))
    save(out / (name + ".command.json"), record)
    with (out / "commands.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if record.get("exit_code") != 0 or record.get("timed_out"):
        raise RuntimeError(f"{name}: exit={record.get('exit_code')} timeout={record.get('timed_out', False)}; see raw logs")
    return record


def logits(path):
    rows = []
    for line in path.read_text().splitlines():
        cols = line.split()
        if len(cols) != 35:
            raise ValueError(f"{path.name}: expected index, token, and 33 logprobs")
        vals = list(map(float, cols[2:]))
        if not all(map(math.isfinite, vals)):
            raise ValueError(f"{path.name}: non-finite logprobs")
        rows.append((int(cols[0]), int(cols[1]), vals))
    return rows


def compare_logits(a, b, atol, rtol):
    if len(a) != len(b) or any(x[:2] != y[:2] for x, y in zip(a, b)):
        return {"passed": False, "token_positions_equal": False, "rows": [len(a), len(b)]}
    errors, failures, changed = [], 0, 0
    for x, y in zip(a, b):
        changed += max(range(33), key=x[2].__getitem__) != max(range(33), key=y[2].__getitem__)
        for p, q in zip(x[2], y[2]):
            e = abs(p - q)
            errors.append(e)
            failures += e > atol + rtol * abs(p)
    errors.sort()
    n = len(errors)
    return {"passed": failures == 0 and changed == 0, "token_positions_equal": True,
            "token_rows": len(a), "values": n, "outside_tolerance": failures,
            "argmax_label_mismatches": changed, "max_abs": max(errors, default=0),
            "mean_abs": statistics.mean(errors) if n else 0,
            "rms": math.sqrt(sum(e * e for e in errors) / n) if n else 0,
            "p99_abs": errors[max(0, math.ceil(n * .99) - 1)] if n else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="JSONL with id and single-line text fields; input and outputs are archived")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--quick", action="store_true", help="first four corpus rows, one timed repetition by default")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8, help="PF_THREADS for both binaries; older binaries may ignore it")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--atol", type=float, default=.001)
    ap.add_argument("--rtol", type=float, default=.0001)
    args = ap.parse_args()
    reps = args.reps if args.reps is not None else (1 if args.quick else 3)
    if (min(args.ctx, args.batch, reps, args.timeout) <= 0 or not 1 <= args.threads <= 64
            or min(args.atol, args.rtol) < 0 or not all(map(math.isfinite, (args.timeout, args.atol, args.rtol)))):
        ap.error("ctx, batch, reps and timeout must be positive; threads 1..64; tolerances nonnegative")
    out = args.results.resolve() / (time.strftime("model-%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    summary = {"schema": 1, "status": "running", "result_dir": str(out), "reps": reps,
               "timing_scope": "CLI end-to-end, startup included; logged warmup before timed runs; page-cache residency not guaranteed",
               "logprob_precision": "pf textual dump rounds to six decimals; argmax labels reconstructed from that dump",
               "maxrss_unit": "KiB on Linux", "atol": args.atol, "rtol": args.rtol, "ctx": args.ctx,
               "batch": args.batch, "threads_requested": args.threads, "quick": args.quick,
               "invocation": sys.argv, "comparisons": {}, "timings": {}}
    save(out / "summary.json", summary)
    try:
        binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
        required = list(binaries.values()) + [args.corpus.resolve()]
        required += [args.model.resolve() / n for n in ("model.safetensors", "o200k_base.tiktoken")]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise RuntimeError("Missing required files: " + ", ".join(missing))
        if any(not os.access(p, os.X_OK) for p in binaries.values()):
            raise RuntimeError("Baseline and candidate must be executable")
        env = {"PF_THREADS": str(args.threads)}
        host = {"platform": platform.platform(), "python": sys.version,
                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "environment": {k: v for k, v in os.environ.items() if k.startswith(("PF_", "OMP_", "MKL_", "OPENBLAS_"))}}
        for p in ("/proc/cpuinfo", "/proc/self/cgroup", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
            if Path(p).is_file():
                host[p] = Path(p).read_text()
        save(out / "host.json", host)
        model_files = required[-2:] + ([args.model.resolve() / "config.json"] if (args.model / "config.json").is_file() else [])
        save(out / "identities.json", {"binaries": {k: identity(v) for k, v in binaries.items()},
                                      "model": [identity(p) for p in model_files], "corpus": identity(args.corpus),
                                      "runner": identity(Path(__file__))})
        rows = [json.loads(s) for s in args.corpus.read_text().splitlines() if s.strip()]
        if args.quick:
            rows = rows[:4]
        if not rows or any(not isinstance(r.get("text"), str) or "\n" in r["text"] for r in rows):
            raise RuntimeError("Corpus requires JSONL rows with single-line string text fields")
        (out / "corpus.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        inp = out / "input.txt"
        inp.write_text("".join(r["text"] + "\n" for r in rows))
        summary["documents"] = len(rows)
        common = ["--model", str(args.model.resolve()), "--ctx", str(args.ctx), "--batch", str(args.batch)]
        def command(kind, name, flags=(), source=inp, packed=True):
            return run(out, name, [str(binaries[kind]), *common, *( ["--lines"] if packed else []),
                                  "-f", str(source), *flags], env, args.timeout)
        modes = {"masks": [], "spans": ["--json"], "tokens": ["--dump-tokens"], "logprobs": ["--dump-logprobs"]}
        for kind in binaries:
            for mode, flags in modes.items():
                command(kind, f"{kind}-{mode}", flags)
        def raw(kind, mode):
            return (out / f"{kind}-{mode}.stdout").read_bytes()
        for mode in ("masks", "tokens"):
            summary["comparisons"][mode + "_exact"] = raw("baseline", mode) == raw("candidate", mode)
        spans = {kind: [json.loads(s) for s in raw(kind, "spans").splitlines()] for kind in binaries}
        summary["comparisons"]["spans_exact"] = spans["baseline"] == spans["candidate"]
        summary["comparisons"]["span_labels_exact"] = [[s["label"] for s in d["spans"]] for d in spans["baseline"]] == [[s["label"] for s in d["spans"]] for d in spans["candidate"]]
        parsed = {kind: logits(out / f"{kind}-logprobs.stdout") for kind in binaries}
        summary["comparisons"]["logprobs"] = compare_logits(parsed["baseline"], parsed["candidate"], args.atol, args.rtol)
        summary["tokens"] = len(parsed["baseline"])
        packing = {}
        for kind in binaries:
            checked = []
            for i, row in enumerate(rows):
                if not row["text"]:  # Empty standalone CLI exits without JSON; packed empties remain covered above.
                    continue
                single = out / f"document-{i:03d}.txt"
                if not single.exists():
                    single.write_text(row["text"])
                name = f"{kind}-separate-{i:03d}"
                command(kind, name, ["--json"], single, False)
                same = json.loads((out / (name + ".stdout")).read_text()) == spans[kind][i]
                checked.append({"document": row["id"], "spans_equal": same})
            packing[kind] = checked
        summary["comparisons"]["packing"] = packing
        ok = all(v for k, v in summary["comparisons"].items() if k.endswith("_exact"))
        ok &= summary["comparisons"]["logprobs"]["passed"] and all(r["spans_equal"] for v in packing.values() for r in v)
        if not ok:
            raise RuntimeError("Regression comparison failed; performance repetitions skipped")
        for kind in binaries:
            command(kind, kind + "-warmup", ["--stats"])
        samples = {kind: [] for kind in binaries}
        for rep in range(reps):
            for kind in (("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")):
                result = command(kind, f"{kind}-timed-{rep:03d}", ["--stats"])
                if (out / f"{kind}-timed-{rep:03d}.stdout").read_bytes() != raw(kind, "masks"):
                    raise RuntimeError(f"{kind}: output changed during timed repetition {rep}")
                samples[kind].append(result)
        for kind, data in samples.items():
            walls = [r["wall_s"] for r in data]
            summary["timings"][kind] = {"samples": data, "wall_median_s": statistics.median(walls),
                                       "wall_min_s": min(walls), "wall_max_s": max(walls),
                                       "tokens_per_s_median": summary["tokens"] / statistics.median(walls)}
        summary["speedup_median"] = summary["timings"]["baseline"]["wall_median_s"] / summary["timings"]["candidate"]["wall_median_s"]
        summary["status"] = "passed"
    except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        summary.update(status="failed", error=str(exc))
    save(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "result_dir": str(out), "error": summary.get("error")}))
    if summary["status"] != "passed":
        print(summary["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
