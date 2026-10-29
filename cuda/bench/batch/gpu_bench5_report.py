"""Gate and report for gpu_bench5.sh (E21: f16_dbuf, stack_fd, stack_fdm vs REF = host_moe_attn_f16, plus the
initial port as BASE).

usage: python -I gpu_bench5_report.py --gate OUT_DIR config [config ...]
         -> one line per config: '<config> PASS|FAIL <span_agree_pct|nan> <mean_dlp|nan> <reasons>'
       python -I gpu_bench5_report.py OUT_DIR WORK_DIR NRUNS NBASE GPU_SECONDS REF BASE config [config ...]
         -> trials_<config>.json (e2e), trials_<config>_forward.json, summary.json  (BASE '-' = none)
A config is a binary plus its args (OUT/configs.txt: '<config> <binary> <args>'). Reads
OUT/check/{ok.txt,gate.txt,identity.txt,<c>.vs_bf16emu.json,<c>.gpumem}, OUT/timing/{<c>.r<i>.{err,md5},
order.txt,timed.txt,rounds_done.txt}, WORK/<binary>/SHA12.
Gate (as gpu_bench4_report.py): built + check runs ok, compare.py vs the bf16-emulating CPU oracle with span
and masked doc agreement >= 98.0% and mean |dlp| <= 0.23. A timed config must also give identical mixed.txt
output md5 across its runs and parse every completed timed round it was scheduled in (BASE: rounds 1..NBASE).
f16_dbuf, stack_fd and stack_fdm are predicted bit-identical to REF: their check.txt outputs (-j, masked,
log-prob dump) and mixed md5 must equal REF's (IDENTITY_REQUIRED); a mismatch is a correctness failure even if
the gate passes. BASE (the initial port) has other numerics, so its identity is informational.
Paired ratios are per round (e2e and forward tok/s, row / column) between every two timed configs over the
rounds both completed; the order is rotated per round, so each pair shares a round but not a slot.
headline = the correct config with the highest median e2e, its paired ratio vs REF and vs BASE.
stacking = each stack's paired ratio vs the best single mechanism (f16_dbuf), the experiment's falsifier.
"""
import json, os, re, statistics as st, sys

MIN_AGREE, MAX_MEAN_DLP = 98.0, 0.23
IDENTITY_REQUIRED = ("f16_dbuf", "stack_fd", "stack_fdm")
BEST_SINGLE, STACKS = "f16_dbuf", ("stack_fd", "stack_fdm")
RE_FWD = re.compile(r"forward ([\d.]+)ms \(([\d.]+) tok/s\)")
RE_E2E = re.compile(r"\| e2e ([\d.]+)ms \(([\d.]+) tok/s\)")
RE_TOK = re.compile(r"([\d]+) docs, (\d+) tok, (\d+) batches")
RE_PIPE = re.compile(r"main pack ([\d.]+)ms forward ([\d.]+)ms scatter ([\d.]+)ms write ([\d.]+)ms wait ([\d.]+)ms")


def parse_stats(text):
    """--stats stderr of the pipelined pf_cuda ('pipelined | ...')."""
    f, e = RE_FWD.search(text), RE_E2E.search(text)
    if not (f and e):
        return None
    r = dict(forward_ms=float(f[1]), forward_tok_s=float(f[2]), e2e_ms=float(e[1]), e2e_tok_s=float(e[2]),
             mode="pipelined" if "pipelined |" in text else "serial")
    t = RE_TOK.search(text)
    if t:
        r.update(docs=int(t[1]), tokens=int(t[2]), batches=int(t[3]))
    p = RE_PIPE.search(text)
    if p:
        r["main_thread_ms"] = dict(zip(("pack", "forward", "scatter", "write", "wait"), map(float, p.groups())))
    for ln in text.splitlines():
        if ln.startswith("gpu phases"):
            r["gpu_phase_ms"] = {k.strip(): float(v) for k, v in
                                 re.findall(r"([a-z0-9+/() -]+?) ([\d.]+) \|", ln.split(":", 1)[1])}
    return r


def summ(v):
    if not v:
        return None
    return dict(median=st.median(v), min=min(v), max=max(v), mean=st.mean(v),
                stdev=st.stdev(v) if len(v) > 1 else 0.0, n=len(v))


def read(path, default=None):
    try:
        return open(path).read().strip()
    except OSError:
        return default


def paired(r, b, metric):
    """per-round ratio r/b over timed rounds both completed (+ the ratios and the rounds, in round order)"""
    bb = {x["round"]: x for x in b["runs"] if not x["warmup"]}
    xs = [x for x in r["runs"] if not x["warmup"] and x["round"] in bb]
    v = [x[metric] / bb[x["round"]][metric] for x in xs]
    s = summ(v)
    if s:
        s.update(ratios=v, rounds=[x["round"] for x in xs], wins=sum(x > 1 for x in v))
    return s


def gate(out, n, ok):
    """-> (passed, compare report or None, reasons)"""
    cmp_path = f"{out}/check/{n}.vs_bf16emu.json"
    rep = None
    if os.path.exists(cmp_path):
        try:
            rep = json.load(open(cmp_path))
        except ValueError:
            rep = None
    why = []
    if ok.get(n) != "1":
        why.append("build or check run failed")
    if not rep:
        why.append("no compare report")
    else:
        if rep["span_agree_pct"] < MIN_AGREE:
            why.append(f"span agree {rep['span_agree_pct']:.2f}% < {MIN_AGREE}")
        if (rep.get("masked_agree_pct") or 0) < MIN_AGREE:
            why.append("masked agree below gate")
        if rep.get("lp_mean_abs_diff") is None or rep["lp_mean_abs_diff"] > MAX_MEAN_DLP:
            why.append(f"mean |dlp| {rep.get('lp_mean_abs_diff')} > {MAX_MEAN_DLP}")
    return not why, rep, why


def read_table(path):
    """whitespace table -> {first column: rest of the columns}"""
    return {ln.split()[0]: ln.split()[1:] for ln in (read(path, "") or "").splitlines() if ln.strip()}


def main_gate(out, names):
    ok = {k: v[0] for k, v in read_table(f"{out}/check/ok.txt").items()}
    for n in names:
        passed, rep, why = gate(out, n, ok)
        span = f"{rep['span_agree_pct']:.4f}" if rep else "nan"
        dlp = f"{rep['lp_mean_abs_diff']:.5f}" if rep and rep.get("lp_mean_abs_diff") is not None else "nan"
        print(n, "PASS" if passed else "FAIL", span, dlp, "; ".join(why))


def main():
    out, work, nruns, nbase, gpu_s, ref, base, names = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                                        int(sys.argv[4]), int(sys.argv[5]), sys.argv[6],
                                                        sys.argv[7], sys.argv[8:])
    base = None if base == "-" else base
    ok = {k: v[0] for k, v in read_table(f"{out}/check/ok.txt").items()}
    cfg = read_table(f"{out}/configs.txt")
    ident = read_table(f"{out}/check/identity.txt")
    rounds_done = int(read(f"{out}/timing/rounds_done.txt", "-1") or -1)
    res = {}
    for n in names:
        binary = (cfg.get(n) or [n])[0]
        args = " ".join((cfg.get(n) or [n, ""])[1:])
        r = dict(name=n, binary=binary, args=args, sha12=read(f"{work}/{binary}/SHA12", "unknown"),
                 built_and_ran=ok.get(n) == "1")
        r["candidate_id"] = f"{binary}@{r['sha12']}" + (f"[{args.replace(' ', '=')}]" if args.startswith("--") else "")
        cmp_path = f"{out}/check/{n}.vs_bf16emu.json"
        gate_ok, rep, why = gate(out, n, ok)
        r["gate_passed"] = gate_ok
        if rep:
            r["vs_bf16emu"] = {k: rep.get(k) for k in ("docs", "span_agree_pct", "masked_agree_pct",
                                                       "lp_mean_abs_diff", "lp_max_abs_diff", "argmax_flips")}
            r["vs_bf16emu"]["span_disagree_ndocs"] = len(rep.get("span_disagree_docs", []))
        mem = [int(x) for x in (read(f"{out}/check/{n}.gpumem", "") or "").split() if x.isdigit()]
        r["check_peak_gpu_mem_mib"] = max(mem) if mem else None
        if n != ref:
            j = ident.get(n)
            r["check_identity_vs_ref"] = dict(zip(("jsonl", "masked", "logprob_dump"), j)) if j else None
        runs = []
        for i in range(nruns + 1):
            s = parse_stats(read(f"{out}/timing/{n}.r{i}.err", "") or "")
            md5 = read(f"{out}/timing/{n}.r{i}.md5")
            if s is not None:
                runs.append(dict(round=i, warmup=i == 0, output_md5=md5, **s))
        pre = parse_stats(read(f"{out}/timing/{n}.pre.err", "") or "")
        if pre:
            r["presweep_run"] = dict(output_md5=read(f"{out}/timing/{n}.pre.md5"), **pre)
        r["runs"] = runs
        r["timed"] = any(os.path.exists(f"{out}/timing/{n}.r{i}.md5") for i in range(nruns + 1))
        timed = [x for x in runs if not x["warmup"]]
        md5s = sorted({read(f"{out}/timing/{n}.r{i}.md5") for i in range(nruns + 1)
                       if os.path.exists(f"{out}/timing/{n}.r{i}.md5")})
        r["output_md5_unique"] = md5s
        if r["timed"]:
            if len(md5s) != 1 or "FAILED" in md5s:
                why.append(f"mixed.txt outputs not identical across runs (or a run failed): {md5s}")
            due = min(nbase, max(rounds_done, 0)) if n == base else max(rounds_done, 0)
            if len(timed) != due or not timed:
                why.append(f"{len(timed)}/{due} completed timed rounds parsed")
        r["correctness_failures"] = why
        r["e2e_tok_s"] = summ([x["e2e_tok_s"] for x in timed])
        r["forward_tok_s"] = summ([x["forward_tok_s"] for x in timed])
        r["e2e_ms"] = summ([x["e2e_ms"] for x in timed])
        if timed and "gpu_phase_ms" in timed[0]:
            r["gpu_phase_ms_median"] = {k: st.median([x["gpu_phase_ms"].get(k, 0.0) for x in timed])
                                        for k in timed[0]["gpu_phase_ms"]}
        if timed and "main_thread_ms" in timed[0]:
            r["main_thread_ms_median"] = {k: st.median([x["main_thread_ms"][k] for x in timed])
                                          for k in timed[0]["main_thread_ms"]}
        res[n] = r

    # identity requirements need REF's mixed md5, so finish correctness after the loop
    ref_md5 = res.get(ref, {}).get("output_md5_unique")
    for n, r in res.items():
        why = r["correctness_failures"]
        if n != ref:
            md5s = r["output_md5_unique"] or ([r["presweep_run"]["output_md5"]] if "presweep_run" in r else [])
            r["mixed_md5_same_as_ref"] = (md5s == ref_md5) if md5s and ref_md5 else None   # None: not compared
            if n in IDENTITY_REQUIRED:
                ci = r.get("check_identity_vs_ref") or {}
                if not ci or any(v != "same" for v in ci.values()):
                    why.append(f"check.txt outputs not identical to {ref}: {ci or 'not compared'}")
                if r["timed"] and r["mixed_md5_same_as_ref"] is not True:
                    why.append(f"mixed.txt md5 differs from {ref}")
        r["correctness_passed"] = r["gate_passed"] and not why
        if r["runs"]:
            for metric, suffix in (("e2e_tok_s", ""), ("forward_tok_s", "_forward")):
                tri = dict(workload_id="mixed_e2e", environment_id="h100", candidate_id=r["candidate_id"],
                           metric=metric, direction="higher", tokens=r["runs"][0].get("tokens"),
                           trials=[dict(pair_id=f"r{x['round']}", value=x[metric], warmup=x["warmup"])
                                   for x in r["runs"]],
                           correctness=dict(passed=r["correctness_passed"],
                                            evidence_path=f"{out}/check/{n}.vs_bf16emu.json", failures=why))
                json.dump(tri, open(f"{out}/trials_{n}{suffix}.json", "w"), indent=1)
        if n != ref and ref in res and r["runs"] and res[ref]["runs"]:
            r["paired_vs_ref"] = dict(vs=ref, **{f"speedup_{m}": paired(r, res[ref], m)
                                                 for m in ("e2e_tok_s", "forward_tok_s")})
        r["paired_e2e_vs"] = {o: paired(r, res[o], "e2e_tok_s") for o in res
                              if o != n and r["runs"] and res[o]["runs"] and paired(r, res[o], "e2e_tok_s")}
    correct = [n for n, r in res.items() if r["correctness_passed"] and r["e2e_tok_s"] and n != base]
    best = max(correct, key=lambda n: res[n]["e2e_tok_s"]["median"]) if correct else None
    headline = None
    if best:
        headline = dict(best=best, candidate_id=res[best]["candidate_id"], e2e_tok_s=res[best]["e2e_tok_s"],
                        paired_vs_ref=res[best]["paired_e2e_vs"].get(ref),
                        paired_vs_base=res[best]["paired_e2e_vs"].get(base) if base else None,
                        base_correct=res[base]["correctness_passed"] if base in res else None)
    stacking = {s: dict(correct=res[s]["correctness_passed"],
                        paired_vs_best_single=res[s]["paired_e2e_vs"].get(BEST_SINGLE),
                        paired_vs_ref=res[s]["paired_e2e_vs"].get(ref))
                for s in STACKS if s in res}
    summary = dict(nruns=nruns, rounds_done=rounds_done, gpu_phase_seconds=gpu_s, ref=ref, base=base,
                   base_rounds=nbase, best_single=BEST_SINGLE, headline=headline, stacking=stacking,
                   identity_required=list(IDENTITY_REQUIRED), order=read(f"{out}/timing/order.txt"),
                   timed_set=read(f"{out}/timing/timed.txt"),
                   gate=dict(min_doc_agree_pct=MIN_AGREE, max_mean_abs_dlp=MAX_MEAN_DLP,
                             oracle="experiments/E1/oracle_bf16emu"),
                   env=read(f"{out}/env.txt"), binaries=res)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(f"{'config':28s} {'gate':5s} {'ok':5s} {'agree%':>7s} {'|dlp|':>6s} {'e2e tok/s (med)':>16s} "
          f"{'x ref (paired)':>15s} {'id/md5':>7s} {'MiB':>6s}  notes")
    for n, r in res.items():
        e, g = r["e2e_tok_s"], r.get("vs_bf16emu") or {}
        sp = ((r.get("paired_vs_ref") or {}).get("speedup_e2e_tok_s"))
        sp = f"{sp['median']:.4f}+-{sp['stdev']:.4f}" if sp else "-"
        ci = r.get("check_identity_vs_ref")
        idm = "ref" if n == ref else ("same" if ci and all(v == "same" for v in ci.values()) else "diff") + \
            {True: "/=", False: "/x", None: "/-"}[r.get("mixed_md5_same_as_ref")]
        ag, dl = g.get("span_agree_pct"), g.get("lp_mean_abs_diff")
        print(f"{n:28s} {str(r['gate_passed']):5s} {str(r['correctness_passed']):5s} "
              f"{ag if ag is not None else float('nan'):7.2f} {dl if dl is not None else float('nan'):6.3f} "
              f"{e['median'] if e else float('nan'):16.0f} {sp:>15s} {idm:>7s} "
              f"{r['check_peak_gpu_mem_mib'] if r['check_peak_gpu_mem_mib'] is not None else '-':>6}  "
              f"{'' if r['timed'] else '[not timed] '}{'; '.join(r['correctness_failures'])}")
    fmt = lambda p: f"{p['median']:.4f} [{p['min']:.4f}, {p['max']:.4f}] wins {p['wins']}/{p['n']}" if p else "-"
    for s_, v in stacking.items():
        print(f"stacking {s_}: correct={v['correct']} x {BEST_SINGLE} {fmt(v['paired_vs_best_single'])}; "
              f"x {ref} {fmt(v['paired_vs_ref'])}")
    if headline:
        print(f"headline {best}: e2e median {headline['e2e_tok_s']['median']:.0f} tok/s; x {ref} "
              f"{fmt(headline['paired_vs_ref'])}; x {base} {fmt(headline['paired_vs_base'])}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--gate":
        main_gate(sys.argv[2], sys.argv[3:])
    else:
        main()
