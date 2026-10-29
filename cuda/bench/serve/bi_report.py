"""Collects serve_bench2.sh's outputs into OUT/results.json and prints a summary.

usage: python3 -I bi_report.py OUT_DIR
Works on partial runs (watchdog, failed build): missing parts are null and listed in "missing".
"""
import glob
import json
import os
import re
import statistics
import sys

MIN_AGREE, MAX_MEAN_DLP = 98.0, 0.23


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return None


def main():
    out = sys.argv[1]
    meta = load(f"{out}/meta.json") or {}
    modes = (meta.get("modes") or "bi default").split()
    res = {"meta": meta, "env": read(f"{out}/env.txt"), "sources_sha12": (read(f"{out}/build/sources.sha12") or "").strip(),
           "missing": []}

    # gemm microbench
    gb = []
    for ln in (read(f"{out}/gemm_bench.jsonl") or "").splitlines():
        try:
            gb.append(json.loads(ln))
        except ValueError:
            pass
    if gb:
        speed = {}
        for x in gb:
            if "bi_over_cublas" in x:
                speed.setdefault(x["shape"], {})[x["M"]] = {"cublas_ms": x["cublas_ms"], "bi_ms": x["bi_ms"],
                                                            "bi_over_cublas": x["bi_over_cublas"]}
        res["gemm_bench"] = {
            "pass": next((x["gemm_bench_pass"] for x in gb if "gemm_bench_pass" in x), None),
            "speed": speed,
            "accuracy": [x for x in gb if "accuracy_pass" in x],
            "invariance": [x for x in gb if "bi_invariant" in x],
            "device": next((x["device"] for x in gb if "device" in x), None),
        }
    else:
        res["gemm_bench"] = None; res["missing"].append("gemm_bench")

    res["invariance"], res["gate"], res["e2e_mixed"], res["server"] = {}, {}, {}, {}
    for m in modes:
        inv = load(f"{out}/inv/{m}.json")
        res["invariance"][m] = inv
        if inv is None:
            res["missing"].append(f"invariance/{m}")

        c = load(f"{out}/check/{m}.vs_bf16emu.json")
        if c is None:
            res["gate"][m] = None; res["missing"].append(f"gate/{m}")
        else:
            why = []
            if c["span_agree_pct"] < MIN_AGREE: why.append("span agree below gate")
            if (c.get("masked_agree_pct") or 0) < MIN_AGREE: why.append("masked agree below gate")
            if c.get("lp_mean_abs_diff") is None or c["lp_mean_abs_diff"] > MAX_MEAN_DLP: why.append("mean |dlp| above gate")
            res["gate"][m] = {"span_agree_pct": c["span_agree_pct"], "masked_agree_pct": c.get("masked_agree_pct"),
                              "lp_mean_abs_diff": c.get("lp_mean_abs_diff"), "lp_max_abs_diff": c.get("lp_max_abs_diff"),
                              "argmax_flips": c.get("argmax_flips"), "passed": not why, "why": why}

        runs = []
        for p in sorted(glob.glob(f"{out}/e2e/{m}.r*.err")):
            r = int(re.search(r"\.r(\d+)\.err$", p).group(1))
            t = read(p) or ""
            g = re.search(r"forward (\d+)ms \(([\d.]+) tok/s\) \| e2e (\d+)ms \(([\d.]+) tok/s\)", t)
            if not g:
                continue
            ph = re.search(r"^gpu phases.*$", t, re.M)
            runs.append({"round": r, "warmup": r == 0, "forward_ms": int(g.group(1)), "forward_tok_s": float(g.group(2)),
                         "e2e_ms": int(g.group(3)), "e2e_tok_s": float(g.group(4)),
                         "md5": (read(p[:-4] + ".md5") or "").strip(), "gpu_phases": ph.group(0) if ph else None})
        timed = [x for x in runs if not x["warmup"]]
        res["e2e_mixed"][m] = {
            "runs": runs,
            "median_e2e_tok_s": statistics.median([x["e2e_tok_s"] for x in timed]) if timed else None,
            "median_forward_tok_s": statistics.median([x["forward_tok_s"] for x in timed]) if timed else None,
            "output_md5_stable": len({x["md5"] for x in runs}) == 1 if runs else None,
        }
        if not timed:
            res["missing"].append(f"e2e/{m}")

        s = load(f"{out}/server/{m}/results.json")
        res["server"][m] = None if s is None else {"summary": s.get("summary"), "agreement": s.get("agreement")}
        if s is None:
            res["missing"].append(f"server/{m}")

    # paired e2e ratio bi / default per timed round
    e = res["e2e_mixed"]
    if all(e.get(m) for m in ("bi", "default")):
        d = {x["round"]: x["e2e_tok_s"] for x in e["default"]["runs"] if not x["warmup"]}
        rat = [x["e2e_tok_s"] / d[x["round"]] for x in e["bi"]["runs"] if not x["warmup"] and x["round"] in d]
        res["e2e_bi_over_default"] = {"per_round": rat, "median": statistics.median(rat) if rat else None,
                                      "min": min(rat) if rat else None, "max": max(rat) if rat else None}
    md = {m: (e.get(m) or {}).get("runs", [{}])[0].get("md5") if (e.get(m) or {}).get("runs") else None for m in modes}
    res["mixed_output_default_eq_bi"] = (md.get("bi") == md.get("default")) if md.get("bi") and md.get("default") else None

    # headline / acceptance
    acc = {}
    for m in modes:
        s = (res["server"].get(m) or {}).get("summary") or {}
        acc[m] = {
            "server": {k: {"docs_s": v["docs_s"], "tok_s": v["tok_s"], "p50_ms": v["p50_ms"], "p99_ms": v["p99_ms"],
                           "agree_pct_vs_batch_path": v["agree_pct_vs_batch_path"], "errors": v["errors"]}
                       for k, v in s.items()},
            "all_runs_agree_pct": ((res["server"].get(m) or {}).get("agreement") or {}).get("all_runs_agree_pct"),
            "exact500": ((res["server"].get(m) or {}).get("agreement") or {}).get("exact_same_batch"),
            "bitwise_invariant": (res["invariance"].get(m) or {}).get("bitwise_invariant"),
            "gate_passed": (res["gate"].get(m) or {}).get("passed"),
            "e2e_median_tok_s": (e.get(m) or {}).get("median_e2e_tok_s"),
        }
    bi = acc.get("bi", {})
    acc["bi_mode_works"] = bool(bi.get("bitwise_invariant") and bi.get("gate_passed")
                                and bi.get("all_runs_agree_pct") == 100.0)
    res["acceptance"] = acc

    with open(f"{out}/results.json", "w") as f:
        f.write(json.dumps(res, indent=1) + "\n")

    # printout
    print(f"results -> {out}/results.json (missing: {', '.join(res['missing']) or 'none'})")
    if res["gemm_bench"]:
        for sh, v in res["gemm_bench"]["speed"].items():
            print(f"gemm {sh:6s} bi/cublas: " + " ".join(f"M{M}={x['bi_over_cublas']:.2f}" for M, x in sorted(v.items(), reverse=True)))
        print(f"gemm_bench pass: {res['gemm_bench']['pass']}")
    for m in modes:
        a = acc[m]
        print(f"[{m}] invariant={a['bitwise_invariant']} gate={a['gate_passed']} "
              f"{res['gate'].get(m)} e2e median={a['e2e_median_tok_s']} server agree={a['all_runs_agree_pct']}")
        for k, v in a["server"].items():
            print(f"   {k:9s} docs/s {v['docs_s']:9.1f} tok/s {v['tok_s']:11.1f} p50 {v['p50_ms']:7.2f} ms "
                  f"p99 {v['p99_ms']:7.2f} ms agree {v['agree_pct_vs_batch_path']}% err {v['errors']}")
    if "e2e_bi_over_default" in res:
        print(f"e2e bi/default per round: {res['e2e_bi_over_default']['per_round']}")
    print(f"bi_mode_works: {acc['bi_mode_works']}")


if __name__ == "__main__":
    main()
