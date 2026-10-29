#!/bin/bash
# Correctness gate for one pf binary (the same runs and limits as gpu_bench5.sh step (ii) + gpu_bench5_report.py --gate):
# check.txt --lines -j / --lines / --lines --dump-logprobs, compare.py vs the bf16-emulating CPU oracle;
# PASS needs span and masked doc agreement >= 98.0% and mean |dlp| <= 0.23.
#   bash final/bench/gate/gate.sh BIN OUT_DIR [extra pf args, e.g. --batch 65536]
# Env: CHECK_TXT (default bench/corpus/check.txt), ORACLE_DIR (default experiments/E1/oracle_bf16emu).
set -u -o pipefail
BIN=${1:?usage: gate.sh BIN OUT_DIR [pf args]}; OUT=${2:?usage: gate.sh BIN OUT_DIR [pf args]}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
[ -n "${CUDA_HOME:-}" ] || . /workspace/env.sh
export PF_MODEL=${PF_MODEL:-/workspace/.cache/pf_model}
CHECK_TXT=${CHECK_TXT:-/workspace/bench/corpus/check.txt}
ORACLE_DIR=${ORACLE_DIR:-/workspace/experiments/E1/oracle_bf16emu}
PY=python3; [ -x /root/.venv/bin/python ] && PY=/root/.venv/bin/python
mkdir -p "$OUT"
"$BIN" --lines -j "$@" -f "$CHECK_TXT" > "$OUT/check.jsonl" 2> "$OUT/run.err" &&
"$BIN" --lines "$@" -f "$CHECK_TXT" > "$OUT/check.masked.txt" 2>> "$OUT/run.err" &&
{ "$BIN" --lines --dump-logprobs "$@" -f "$CHECK_TXT" 2>> "$OUT/run.err" | gzip -1 > "$OUT/check.lp.gz"; } ||
    { echo "FAIL: pf run failed (see $OUT/run.err)"; exit 1; }
$PY -I "$HERE/compare.py" --min-agree 98.0 \
    --ref-jsonl "$ORACLE_DIR/check.jsonl" --cand-jsonl "$OUT/check.jsonl" \
    --ref-masked "$ORACLE_DIR/check.masked.txt" --cand-masked "$OUT/check.masked.txt" \
    --ref-lp "$ORACLE_DIR/check.lp.gz" --cand-lp "$OUT/check.lp.gz" --out "$OUT/vs_oracle.json" > "$OUT/compare.txt"
$PY -I - "$OUT/vs_oracle.json" <<'EOF' | tee "$OUT/gate.txt"
import json, sys
r = json.load(open(sys.argv[1]))
why = [f"{k} {r[k]:.2f}% < 98.0" for k in ("span_agree_pct", "masked_agree_pct") if r.get(k, 0) < 98.0]
if r.get("lp_mean_abs_diff", 9e9) > 0.23: why.append(f"mean |dlp| {r.get('lp_mean_abs_diff')} > 0.23")
print("PASS" if not why else "FAIL", f"agree {r['span_agree_pct']:.4f}% masked {r.get('masked_agree_pct', 0):.4f}%",
      f"mean|dlp| {r.get('lp_mean_abs_diff', float('nan')):.5f} flips {r.get('argmax_flips')}", "; ".join(why))
sys.exit(1 if why else 0)
EOF
