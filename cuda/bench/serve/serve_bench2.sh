#!/bin/bash
# E22: scripted GPU run for candidates/serve2 (pf --serve on the f16_fused engine), in two builds:
#   default  PF_BATCH_INVARIANT=0 (cuBLAS dense GEMMs, f16_fused attention)
#   bi       PF_BATCH_INVARIANT=1 (gemm_bi.cu fixed-tile GEMMs + ATTN_BI attention, no cuBLAS)
# Non-interactive; a watchdog kills everything after SB_MAX_S (540 s), and a soft budget
# (SB_SOFT_S, 500 s) skips server runs that would not finish, so results.json is always written.
#
#   bash /workspace/final/bench/serve/serve_bench2.sh [OUT]   # GPU run, ~6.5 min on an H100; default OUT = final/runs/serve
# (final/ copy of experiments/E22/serve_bench2.sh: server sources from final/server, helper scripts from this folder)
#
# Steps:
#   (1) build both modes + the GEMM microbench (parallel, /tmp, -march=native)
#   (2) gemm_bench: gemm_bi vs cuBLAS on qkv/out/router shapes (speed, accuracy, row invariance)
#   (3) batch-invariance test (bi_invariance.py): 50 docs in 4 packings, raw fp32 log-prob bits;
#       must be bitwise identical for the bi build (informational for default)
#   (4) check.txt gate per build vs experiments/E1/oracle_bf16emu (compare.py, unchanged limits:
#       span and masked doc agreement >= 98.0%, mean |dlp| <= 0.23), compare runs in the background
#   (5) batch-mode E2E on mixed.txt: 1 warmup + SB_E2E_ROUNDS (3) rounds per build, order alternated;
#       the first timed round's output is that build's batch-path reference for the server checks
#   (6) server load matrix per build (bi first): exact500 before/after, c1_short, c1, c8, c64, c256,
#       b16x64 (keep-alive), every response checked against that build's own --lines output
#   (7) bi_report.py -> OUT/results.json
#
# Knobs: SB_DURATION (12 s per load run), SB_WARMUP (2 s), SB_PROCS (4), SB_CONCS ("1 8 64 256"),
#   SB_E2E_ROUNDS (3), SB_MAX_S (540), SB_SOFT_S (500), SB_BIN (pf_cuda; pf_cpu for a CPU dry run),
#   SB_CORPUS, SB_CHECK, SB_LINGER (8 ms), SB_MODES ("bi default"), SRC (candidates/serve2).
# PF_NULL_FORWARD=1 is passed through to every pf process (CPU dry run of the whole script).
set -u -o pipefail
if [ -z "${SB_INNER:-}" ]; then
    export SB_INNER=1
    exec timeout -k 15 "${SB_MAX_S:-540}" bash "$0" "$@"
fi

E=$(cd "$(dirname "$0")" && pwd); FINAL=$(cd "$E/../.." && pwd)
SRC=${SRC:-$FINAL/server}
LG="python3 -I $E/loadgen.py"
CORPUS=${SB_CORPUS:-/workspace/bench/corpus/mixed.txt}
CHECK_TXT=${SB_CHECK:-/workspace/bench/corpus/check.txt}
ORACLE_DIR=${ORACLE_DIR:-/workspace/experiments/E1/oracle_bf16emu}
DUR=${SB_DURATION:-12}; WARM=${SB_WARMUP:-2}; PROCS=${SB_PROCS:-4}
CONCS=${SB_CONCS:-"1 8 64 256"}
BIN_NAME=${SB_BIN:-pf_cuda}
LINGER=${SB_LINGER:-8}
NR=${SB_E2E_ROUNDS:-3}
SOFT=${SB_SOFT_S:-500}
MODES=${SB_MODES:-"bi default"}
OUT=${1:-$FINAL/runs/serve}
[ -e "$OUT" ] && OUT="${OUT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"/{build,inv,check,e2e,server}
LOG="$OUT/bench.log"
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
log() { echo "[$(date +%H:%M:%S) +$(el)s] $*" | tee -a "$LOG"; }
SPID=""
stop_server() {
    if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
        kill "$SPID" 2>/dev/null
        for _ in $(seq 50); do kill -0 "$SPID" 2>/dev/null || break; sleep 0.1; done
        kill -9 "$SPID" 2>/dev/null
    fi
    SPID=""
}
report() {
    python3 -I "$E/bi_report.py" "$OUT" > "$OUT/report.txt" 2>&1; cat "$OUT/report.txt" | tee -a "$LOG"
}
trap 'stop_server' EXIT
trap 'log "terminated (watchdog or signal)"; stop_server; report; exit 124' TERM INT

[ -f /workspace/env.sh ] && . /workspace/env.sh
export PF_MODEL=${PF_MODEL:-/workspace/.cache/pf_model}
PY=python3; [ -x /root/.venv/bin/python ] && PY=/root/.venv/bin/python
for f in "$CORPUS" "$CHECK_TXT" "$SRC/pf.c" "$SRC/gemm_bi.cu" "$PF_MODEL" "$E/bi_invariance.py" "$E/bi_report.py" \
         "$E/gemm_bench/gemm_bench.cu" "$E/loadgen.py" "$FINAL/bench/gate/compare.py" \
         "$ORACLE_DIR/check.jsonl" "$ORACLE_DIR/check.masked.txt" "$ORACLE_DIR/check.lp.gz"; do
    [ -e "$f" ] || { log "missing input $f"; exit 2; }
done
$PY -I -c 'import numpy' 2>/dev/null || { log "$PY has no numpy (compare.py needs it)"; exit 2; }
GPU=0; nvidia-smi -L > /dev/null 2>&1 && [ "$BIN_NAME" = pf_cuda ] && GPU=1
{
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi"
    echo "nproc $(nproc) | python $(python3 --version 2>&1) | PF_NULL_FORWARD=${PF_NULL_FORWARD:-unset}"
    echo "bin $BIN_NAME | modes $MODES | load runs ${DUR}s + ${WARM}s warmup | loadgen procs $PROCS | linger ${LINGER}ms | e2e rounds $NR"
} > "$OUT/env.txt"
log "out $OUT"; tee -a "$LOG" < "$OUT/env.txt"

# ---- (1) builds ---------------------------------------------------------------------------------
W=/tmp/pfserve2_$$
rm -rf "$W"; mkdir -p "$W"
declare -A BI_FLAG=([default]=0 [bi]=1) BIN OKB
for m in $MODES; do
    mkdir -p "$W/$m"; cp "$SRC"/*.c "$SRC"/*.cu "$SRC"/*.h "$SRC"/*.inc "$SRC"/build.sh "$W/$m"/
    ( PF_BATCH_INVARIANT=${BI_FLAG[$m]} timeout 300 sh "$W/$m/build.sh" ) > "$OUT/build/$m.log" 2>&1 &
    eval "BP_$m=$!"
done
if [ "$GPU" = 1 ]; then
    mkdir -p "$W/gb"
    ( cd "$W/gb" && timeout 300 nvcc -O3 -arch=sm_90a -std=c++17 -I"$SRC" "$E/gemm_bench/gemm_bench.cu" \
        "$SRC/gemm_bi.cu" -lcublas -o gemm_bench ) > "$OUT/build/gemm_bench.log" 2>&1 &
    GBP=$!
fi
SHA=$(cat "$SRC"/pf.c "$SRC"/gpu.cu "$SRC"/attn.cu "$SRC"/gemm_bi.cu "$SRC"/moe.cu | sha256sum | cut -c1-12)
for m in $MODES; do
    eval "p=\$BP_$m"
    if wait "$p" && [ -x "$W/$m/$BIN_NAME" ]; then OKB[$m]=1; BIN[$m]=$W/$m/$BIN_NAME; log "build $m ok"
    else OKB[$m]=0; log "build $m FAILED"; tail -n 15 "$OUT/build/$m.log" | tee -a "$LOG"; fi
done
echo "$SHA" > "$OUT/build/sources.sha12"
log "sources sha12 $SHA"

# ---- (2) GEMM microbench ------------------------------------------------------------------------
if [ "$GPU" = 1 ]; then
    if wait "$GBP" && [ -x "$W/gb/gemm_bench" ]; then
        timeout 120 "$W/gb/gemm_bench" > "$OUT/gemm_bench.jsonl" 2> "$OUT/gemm_bench.err"
        log "gemm_bench exit $? ($(grep -c . "$OUT/gemm_bench.jsonl") lines)"
        grep -h '"bi_over_cublas"\|bi_invariant\|accuracy_pass\|gemm_bench_pass' "$OUT/gemm_bench.jsonl" | tee -a "$LOG"
    else log "gemm_bench build FAILED"; fi
else
    log "gemm_bench skipped (no GPU or SB_BIN != pf_cuda)"
fi

# ---- (3) tokens per line + invariance test --------------------------------------------------------
ANY=""; for m in $MODES; do [ "${OKB[$m]}" = 1 ] && { ANY=${BIN[$m]}; break; }; done
[ -n "$ANY" ] || { log "no build succeeded"; report; exit 3; }
timeout 120 "$ANY" --lines --dump-ntok -f "$CORPUS" > "$OUT/ntok.txt" 2> "$OUT/ntok.err" || { log "dump-ntok failed"; report; exit 3; }
NDOC=$(wc -l < "$OUT/ntok.txt"); NTOK=$(awk '{s+=$1} END {print s}' "$OUT/ntok.txt")
log "corpus: $NDOC docs, $NTOK tokens"
$LG sample --corpus "$CORPUS" --n 500 --out-txt "$OUT/check500.txt" --out-idx "$OUT/check500.idx" | tee -a "$LOG"
for m in $MODES; do
    [ "${OKB[$m]}" = 1 ] || continue
    timeout 200 python3 -I "$E/bi_invariance.py" "${BIN[$m]}" "$CORPUS" "$OUT/ntok.txt" "$OUT/inv" --label "$m" \
        2>> "$OUT/inv/$m.err" | tee -a "$LOG" || log "invariance test $m failed to run (see inv/$m.err)"
    rm -f "$OUT"/inv/"$m"_?.txt
done

# ---- (4) check.txt gate runs (compare in background) ------------------------------------------------
CMP=()
for m in $MODES; do
    [ "${OKB[$m]}" = 1 ] || continue
    f=$OUT/check/$m b=${BIN[$m]}
    if timeout 150 "$b" --lines -j -f "$CHECK_TXT" > "$f.jsonl" 2> "$f.jsonl.err" &&
       timeout 150 "$b" --lines -f "$CHECK_TXT" > "$f.masked.txt" 2> "$f.masked.err" &&
       timeout 150 "$b" --lines --dump-logprobs -f "$CHECK_TXT" 2> "$f.lp.err" | gzip -1 > "$f.lp.gz"; then
        $PY -I "$FINAL/bench/gate/compare.py" --min-agree 98.0 \
            --ref-jsonl "$ORACLE_DIR/check.jsonl" --cand-jsonl "$f.jsonl" \
            --ref-masked "$ORACLE_DIR/check.masked.txt" --cand-masked "$f.masked.txt" \
            --ref-lp "$ORACLE_DIR/check.lp.gz" --cand-lp "$f.lp.gz" \
            --out "$f.vs_bf16emu.json" > "$f.vs_bf16emu.txt" 2>&1 &
        CMP+=($!)
        log "check runs $m ok; compare started"
    else
        log "check runs $m FAILED (see check/$m.*.err)"
    fi
done
if [ -f "$OUT/check/default.masked.txt" ] && [ -f "$OUT/check/bi.masked.txt" ]; then
    cmp -s "$OUT/check/default.masked.txt" "$OUT/check/bi.masked.txt" && s=same || s=DIFF
    log "check.txt masked output default vs bi: $s"
fi

# ---- (5) batch-mode E2E on mixed.txt --------------------------------------------------------------
for r in $(seq 0 "$NR"); do
    if [ $(( r % 2 )) = 0 ]; then ORD="$MODES"; else ORD=$(echo "$MODES" | awk '{for (i = NF; i > 0; i--) printf "%s ", $i}'); fi
    for m in $ORD; do
        [ "${OKB[$m]}" = 1 ] || continue
        o=$OUT/e2e/$m.r$r
        if timeout 150 "${BIN[$m]}" --lines --stats -f "$CORPUS" > "$o.out" 2> "$o.err"; then
            md5sum < "$o.out" | cut -c1-32 > "$o.md5"
            log "e2e $m r$r: $(grep -h '^pipelined' "$o.err" | sed 's/.*e2e/e2e/') md5 $(cat "$o.md5")"
            if [ "$r" = 1 ]; then mv "$o.out" "$OUT/server/ref_corpus.$m.masked.txt"; else rm -f "$o.out"; fi
        else
            log "e2e $m r$r FAILED"
        fi
    done
done

# ---- (6) server load matrix per build ---------------------------------------------------------------
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
start_server() {   # $1 = binary, $2 = log
    "$1" --serve "$PORT" --stats --linger "$LINGER" > "$2" 2>&1 &
    SPID=$!
    for _ in $(seq 240); do
        python3 -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=1)" \
            2>/dev/null && { log "server up (pid $SPID, port $PORT)"; return 0; }
        kill -0 "$SPID" 2>/dev/null || break
        sleep 0.5
    done
    log "server did not come up"; tail -n 20 "$2" | tee -a "$LOG"; return 1
}
for m in $MODES; do
    [ "${OKB[$m]}" = 1 ] || continue
    S=$OUT/server/$m; mkdir -p "$S/runs"
    REF=$OUT/server/ref_corpus.$m.masked.txt
    if [ ! -s "$REF" ]; then
        timeout 150 "${BIN[$m]}" --lines -f "$CORPUS" > "$REF" 2> "$S/ref_corpus.err" || { log "ref corpus $m failed"; continue; }
    fi
    timeout 120 "${BIN[$m]}" --lines -f "$OUT/check500.txt" > "$S/ref500.masked.txt" 2> "$S/ref500.err" \
        || { log "ref500 $m failed"; continue; }
    [ $(el) -lt $(( SOFT - 30 )) ] || { log "skip server $m (soft budget)"; continue; }
    start_server "${BIN[$m]}" "$S/server.log" || { stop_server; continue; }
    JS=()
    DEAD=0
    run_load() {   # $1 = name, rest = loadgen run args
        local name=$1; shift
        [ "$DEAD" = 1 ] && return 1
        local d=$DUR; [ "$name" = c1_short ] && d=$(( (${DUR%.*} + 1) / 2 ))
        if [ $(( $(el) + d + ${WARM%.*} + 12 )) -gt "$SOFT" ]; then log "skip $m/$name (soft budget)"; return 1; fi
        local l0; l0=$(wc -l < "$S/server.log")
        timeout $(( d + ${WARM%.*} + 60 )) $LG run --port "$PORT" --corpus "$CORPUS" --ntok "$OUT/ntok.txt" \
            --ref "$REF" --name "$name" --warmup "$WARM" --procs "$PROCS" --duration "$d" \
            --out "$S/runs/$name.json" "$@" 2>> "$LOG" | sed "s/^/$m /" | tee -a "$LOG"
        [ -f "$S/runs/$name.json" ] && JS+=("$S/runs/$name.json")
        tail -n +"$(( l0 + 1 ))" "$S/server.log" > "$S/runs/$name.server.log"
        kill -0 "$SPID" 2>/dev/null || { log "server $m died"; tail -n 5 "$S/server.log" | tee -a "$LOG"; DEAD=1; return 1; }
    }
    exact() {
        [ "$DEAD" = 1 ] && return 1
        timeout 120 $LG exact --port "$PORT" --docs "$OUT/check500.txt" --ref "$S/ref500.masked.txt" \
            --name "$1" --out "$S/runs/$1.json" 2>> "$LOG" | sed "s/^/$m /" | tee -a "$LOG"
        [ -f "$S/runs/$1.json" ] && JS+=("$S/runs/$1.json")
    }
    exact exact500_before
    run_load c1_short --keepalive --conc 1 --max-tok 64
    for c in $CONCS; do run_load "c$c" --keepalive --conc "$c"; done
    run_load b16x64 --keepalive --conc 16 --batch-docs 64
    exact exact500_after
    stop_server
    [ "${#JS[@]}" -gt 0 ] && $LG report "${JS[@]}" --out "$S/results.json" | sed "s/^/$m /" | tee -a "$LOG"
done

# ---- (7) report -----------------------------------------------------------------------------------
for p in "${CMP[@]}"; do wait "$p"; done
python3 - "$OUT/meta.json" <<EOF
import json, sys
json.dump({"candidate": "serve2@$SHA", "binary": "$BIN_NAME", "modes": "$MODES", "corpus": "$CORPUS",
           "corpus_docs": $NDOC, "corpus_tokens": $NTOK, "duration_s": $DUR, "warmup_s": $WARM,
           "loadgen_procs": $PROCS, "linger_ms": $LINGER, "e2e_rounds": $NR,
           "null_forward": "${PF_NULL_FORWARD:-}", "env": open("$OUT/env.txt").read(),
           "elapsed_s": $(el)}, open(sys.argv[1], "w"), indent=1)
EOF
report
rm -rf "$W"
log "done in $(el)s"
exit 0
