"""Summarize candidates/scale.sh runs -> OUT/scale_summary.json (see scale.sh header)."""
import glob, hashlib, json, os, re, statistics as st, sys

out = sys.argv[1]
tokens = int(os.environ.get("TOKENS", "7371164"))
rows = [l.split() for l in open(os.path.join(out, "runs.txt")) if l.strip()]


def proc_e2e_ms(path):
    ms = re.findall(r"e2e ([0-9.]+)\s*ms", open(path, errors="replace").read())
    return float(ms[-1]) if ms else None


res = {}
for N, tag, wall, rc in rows:
    N, d = int(N), os.path.join(out, f"n{N}")
    per = [proc_e2e_ms(f) for f in sorted(glob.glob(os.path.join(d, f"{tag}.p*.err")))]
    mx = max(per) if per and all(per) else None
    md5 = hashlib.md5(open(os.path.join(d, f"{tag}.out.txt"), "rb").read()).hexdigest()
    res.setdefault(N, {"runs": []})["runs"].append({
        "tag": tag, "warmup": tag == "r0", "rc": int(rc), "wall_ms": int(wall),
        "proc_e2e_ms": per, "max_proc_e2e_ms": mx, "md5": md5,
        "agg_e2e_tok_s": tokens / (mx / 1e3) if mx else None,
        "agg_wall_tok_s": tokens / (int(wall) / 1e3) if int(wall) else None})

for N, r in res.items():
    timed = [x for x in r["runs"] if not x["warmup"] and x["rc"] == 0]
    for k in ("agg_e2e_tok_s", "agg_wall_tok_s"):
        v = [x[k] for x in timed if x[k]]
        r[k + "_median"] = st.median(v) if v else None
        r[k + "_range"] = [min(v), max(v)] if v else None
    r["md5s"] = sorted({x["md5"] for x in r["runs"]})

if 1 in res:
    ref = open(os.path.join(out, "n1", res[1]["runs"][-1]["tag"] + ".out.txt"), "rb").read().split(b"\n")
    base = res[1].get("agg_e2e_tok_s_median")
    for N, r in res.items():
        L = open(os.path.join(out, f"n{N}", r["runs"][-1]["tag"] + ".out.txt"), "rb").read().split(b"\n")
        r["n_lines"] = len(L)
        r["doc_agree_vs_n1_pct"] = 100.0 * sum(a == b for a, b in zip(ref, L)) / max(len(ref), len(L))
        if base and r.get("agg_e2e_tok_s_median"):
            r["scaling_vs_n1"] = r["agg_e2e_tok_s_median"] / base

json.dump({str(k): v for k, v in sorted(res.items())}, open(os.path.join(out, "scale_summary.json"), "w"), indent=1)
for N, r in sorted(res.items()):
    print(f"N={N}: agg_e2e={r.get('agg_e2e_tok_s_median')} tok/s range={r.get('agg_e2e_tok_s_range')} "
          f"agg_wall={r.get('agg_wall_tok_s_median')} scaling={r.get('scaling_vs_n1')} "
          f"agree_vs_n1={r.get('doc_agree_vs_n1_pct')}% md5s={r['md5s']}")
