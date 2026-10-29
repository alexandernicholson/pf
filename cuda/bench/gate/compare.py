"""Correctness gate: compare a candidate pf run against the CPU oracle.

Inputs are pf output files for the same --lines corpus:
  --ref-jsonl / --cand-jsonl    pf --lines -j          (one JSON span list per doc)
  --ref-masked / --cand-masked  pf --lines             (masked text, one line per doc)
  --ref-lp / --cand-lp          pf --lines --dump-logprobs (optional, .gz ok)
Prints a summary and writes --out JSON. Exit 1 if doc agreement < --min-agree.
"""
import argparse, gzip, json, sys
import numpy as np


def lines(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="surrogateescape", newline="\n") as f:
        return f.read().split("\n")[:-1]


def load_lp(path):
    """-> list of per-doc arrays [n_tok, 33] (doc boundary = token index resets to 0)."""
    op = gzip.open if path.endswith(".gz") else open
    docs, cur = [], []
    with op(path, "rt") as f:
        for ln in f:
            p = ln.split()
            if p[0] == "0" and cur:
                docs.append(np.array(cur, dtype=np.float32)); cur = []
            cur.append([float(x) for x in p[2:]])
    if cur:
        docs.append(np.array(cur, dtype=np.float32))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-jsonl", required=True); ap.add_argument("--cand-jsonl", required=True)
    ap.add_argument("--ref-masked"); ap.add_argument("--cand-masked")
    ap.add_argument("--ref-lp"); ap.add_argument("--cand-lp")
    ap.add_argument("--out"); ap.add_argument("--min-agree", type=float, default=100.0)
    a = ap.parse_args()
    rep = {}
    rj, cj = lines(a.ref_jsonl), lines(a.cand_jsonl)
    if len(rj) != len(cj):
        print(f"FAIL: doc count differs ref={len(rj)} cand={len(cj)}"); sys.exit(1)
    bad = [i for i, (x, y) in enumerate(zip(rj, cj)) if x != y]
    rep["docs"] = len(rj)
    rep["span_disagree_docs"] = bad
    rep["span_agree_pct"] = 100.0 * (len(rj) - len(bad)) / max(1, len(rj))
    if a.ref_masked and a.cand_masked:
        rm, cm = lines(a.ref_masked), lines(a.cand_masked)
        mb = [i for i, (x, y) in enumerate(zip(rm, cm)) if x != y] + \
             list(range(min(len(rm), len(cm)), max(len(rm), len(cm))))
        rep["masked_disagree_docs"] = mb
        rep["masked_agree_pct"] = 100.0 * (len(rm) - len(mb)) / max(1, len(rm))
    if a.ref_lp and a.cand_lp:
        rl, cl = load_lp(a.ref_lp), load_lp(a.cand_lp)
        assert len(rl) == len(cl), (len(rl), len(cl))
        mx, sm, n, flips = 0.0, 0.0, 0, []
        for d, (r, c) in enumerate(zip(rl, cl)):
            assert r.shape == c.shape, (d, r.shape, c.shape)
            df = np.abs(r - c)
            mx = max(mx, float(df.max())); sm += float(df.sum()); n += df.size
            ra, ca = r.argmax(1), c.argmax(1)
            for t in np.nonzero(ra != ca)[0]:
                s = np.sort(r[t])
                flips.append({"doc": d, "tok": int(t), "ref": int(ra[t]), "cand": int(ca[t]),
                              "ref_margin": float(s[-1] - s[-2])})
        rep["lp_max_abs_diff"] = mx
        rep["lp_mean_abs_diff"] = sm / max(1, n)
        rep["argmax_flips"] = len(flips)
        rep["argmax_flip_detail"] = flips[:200]
        rep["argmax_flip_max_ref_margin"] = max([f["ref_margin"] for f in flips], default=0.0)
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 20 else f"[{len(v)} items]")
                      for k, v in rep.items() if k != "argmax_flip_detail"}, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    ok = rep["span_agree_pct"] >= a.min_agree
    print("PASS" if ok else "FAIL", f"span agreement {rep['span_agree_pct']:.3f}% (min {a.min_agree})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
