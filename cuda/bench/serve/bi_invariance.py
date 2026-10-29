"""Batch-invariance test for a pf_cuda build (run on the GPU box by serve_bench2.sh).

The same 50 target documents are packed in 4 layouts, with different orders, offsets in the
packed stream, co-batched filler documents and batch sizes. Their log-probs are dumped as raw
fp32 bits (pf --dump-lpbits) and must be bitwise identical to layout A for every target doc.

  A  targets in corpus order, alone, default --batch
  B  targets shuffled, interleaved with fillers (random insertion points), default --batch
  C  fillers first (an odd number of tokens), then targets reversed, --batch 8192 (several small batches)
  D  each target after a different filler, --batch 4096 (= n_ctx, the smallest pf allows: a few docs
     per forward, small M)
Targets: 49 docs spread over the length distribution (1 .. 4096 tokens) plus one longer doc
(pf splits it into 4096-token segments, which land in different batches in C and D).

usage: python3 -I bi_invariance.py BIN CORPUS NTOK OUT_DIR [--label NAME] [--timeout S]
writes OUT_DIR/<label>.json and prints a one-line summary; exit 0 unless a run fails.
"""
import json
import os
import random
import subprocess
import sys


def main():
    a = sys.argv[1:]
    label, timeout = "build", 120
    if "--label" in a:
        i = a.index("--label"); label = a[i + 1]; del a[i:i + 2]
    if "--timeout" in a:
        i = a.index("--timeout"); timeout = float(a[i + 1]); del a[i:i + 2]
    binp, corpus, ntokf, out = a
    os.makedirs(out, exist_ok=True)
    data = open(corpus, "rb").read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    ntok = [int(x) for x in open(ntokf).read().split()]
    if len(ntok) != len(lines):
        sys.exit(f"ntok has {len(ntok)} lines, corpus {len(lines)}")
    rng = random.Random(22)
    # targets: 49 docs spread over the length distribution (1 token .. longest <= 4096) + one longer
    cand = sorted((n, i) for i, n in enumerate(ntok) if 0 < n <= 4096)
    tgt = {cand[min(len(cand) - 1, int(len(cand) * (k + 0.5) / 47))][1] for k in range(47)} | {cand[0][1], cand[-1][1]}
    longer = sorted((abs(n - 6000), i) for i, n in enumerate(ntok) if 4096 < n <= 12000)
    if longer:
        tgt.add(longer[0][1])
    tgt = sorted(tgt)
    pool = [i for i, n in enumerate(ntok) if 0 < n <= 1500 and i not in set(tgt)]
    fill = rng.sample(pool, 400)

    lay = {}
    lay["A"] = (list(tgt), [])
    b = [("t", i) for i in rng.sample(tgt, len(tgt))]
    for f in fill[:200]:
        b.insert(rng.randrange(len(b) + 1), ("f", f))
    lay["B"] = ([i for _, i in b], [])
    c = list(fill[200:330])
    if sum(ntok[i] for i in c) % 2 == 0:
        c.append(fill[330])
    lay["C"] = (c + tgt[::-1], ["--batch", "8192"])
    d = []
    for k, t in enumerate(tgt):
        d += [fill[331 + (k % 69)], t]
    lay["D"] = (d, ["--batch", "4096"])

    res = {"label": label, "bin": binp, "targets": len(tgt),
           "target_tokens": sum(ntok[i] for i in tgt), "layouts": {}}
    bits = {}
    for name, (docs, extra) in lay.items():
        txt = os.path.join(out, f"{label}_{name}.txt")
        with open(txt, "wb") as f:
            f.write(b"\n".join(lines[i] for i in docs) + b"\n")
        cmd = [binp, "--lines", "--dump-lpbits", "-f", txt] + extra
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode != 0:
            print(p.stderr.decode()[-2000:], file=sys.stderr)
            sys.exit(f"{label} layout {name}: exit {p.returncode}")
        rows = p.stdout.split(b"\n")
        if rows and rows[-1] == b"":
            rows.pop()
        if len(rows) != sum(ntok[i] for i in docs):
            sys.exit(f"{label} layout {name}: {len(rows)} dump rows, expected {sum(ntok[i] for i in docs)}")
        per, r = {}, 0
        for i in docs:
            if i in set(tgt):
                per[i] = rows[r:r + ntok[i]]
            r += ntok[i]
        bits[name] = per
        res["layouts"][name] = {"docs": len(docs), "tokens": len(rows), "extra_args": extra}

    def f32(h):
        import struct
        return struct.unpack("<f", int(h, 16).to_bytes(4, "little"))[0]

    all_ok = True
    for name in ("B", "C", "D"):
        same, tok_diff, maxd = 0, 0, 0.0
        for i in tgt:
            ra, rb = bits["A"][i], bits[name][i]
            if ra == rb:
                same += 1
                continue
            for x, y in zip(ra, rb):
                if x != y:
                    tok_diff += 1
                    xs, ys = x.split()[2:], y.split()[2:]
                    maxd = max(maxd, max(abs(f32(p) - f32(q)) for p, q in zip(xs, ys)))
        ok = same == len(tgt)
        all_ok &= ok
        res["layouts"][name].update(docs_identical_to_A=same, target_docs=len(tgt),
                                    tokens_differing=tok_diff, max_abs_lp_diff=maxd, identical=ok)
    res["bitwise_invariant"] = all_ok
    json.dump(res, open(os.path.join(out, f"{label}.json"), "w"), indent=1)
    print(f"invariance {label}: " + " | ".join(
        f"{n}: {res['layouts'][n]['docs_identical_to_A']}/{len(tgt)} docs identical, "
        f"{res['layouts'][n]['tokens_differing']} tok differ, max |dlp| {res['layouts'][n]['max_abs_lp_diff']:.3g}"
        for n in ("B", "C", "D")) + f" -> bitwise_invariant={all_ok}")


if __name__ == "__main__":
    main()
