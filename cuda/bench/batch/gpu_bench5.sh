#!/bin/bash
# Scripted H100 run for E21: build the initial port, REF and three stacked candidates, correctness gate for every
# binary FIRST, bit-identity of the candidates vs REF, then order-rotated paired timing. Non-interactive;
# budgeted to < 9 min of GPU time.
#
#   bash /workspace/final/bench/batch/gpu_bench5.sh [OUT_DIR]   # default OUT = final/runs/gpu_bench5
# (final/ copy of candidates/gpu_bench5.sh: stack_fdm is built from final/engine, compare.py and the report come
#  from final/bench; REF/f16_dbuf/stack_fd and the initial port still build from /workspace/candidates and
#  /workspace/implementation, which are needed only for the paired ratios. NAMES="stack_fdm" REF=stack_fdm BASE=
#  times the final engine alone.)
#
# Binaries (NAMES, all at the default --batch 131072):
#   host_moe_attn_f16  REF (host_moe + tensor-core attention, fp16 q/k/v/P, Variant A)
#   f16_dbuf           REF + double-buffered forward (gpu_submit/gpu_wait, D2H on a copy stream); the kept best
#   stack_fd           f16_dbuf + f16_fused (fused qkv epilogue -> fp16, attn_fwd_f16 Variant B)
#   stack_fdm          stack_fd + MoE v2 (E15 warp-specialized persistent TMA kernel, CfgWS<160,6,160,6,true>)
#   all three predicted byte-identical to REF
# BASE = initial (/workspace/implementation/pf_cuda, the first port, serial forward, ~16 s per mixed.txt run):
#   gated like the others (its numerics differ from REF, so no identity requirement) and timed in rounds 1..NBASE
#   only, inserted into the rotated order, for the end-to-end ratio vs the initial port.
# Steps:
#   (i)   copy each source tree to $WORK/<name> and build it there (in parallel; -march=native)
#   (ii)  check.txt per config: --lines -j, --lines (masked), --lines --dump-logprobs (GPU memory.used is
#         sampled during the -j run); compare.py vs experiments/E1/oracle_bf16emu runs on the CPU in the
#         background. Gate (gpu_bench5_report.py --gate, unchanged limits): span and masked doc agreement
#         >= 98.0% and mean |dlp| <= 0.23 -> OUT/check/gate.txt. Identity of every config vs REF on the three
#         check outputs -> OUT/check/identity.txt (required for f16_dbuf, stack_fd, stack_fdm)
#   (iii) timing set = REF + gate-passing NAMES; round 0 = warmup, rounds 1..N timed, order rotated per round;
#         BASE joins rounds 1..NBASE (gate-passing); a time guard skips rounds that would end past BUDGET_S;
#         a failed/hung run drops that config. Mixed output md5 per run (identity vs REF in the report)
#   (iv)  trials_<config>.json (e2e) + trials_<config>_forward.json + summary.json via gpu_bench5_report.py
#         (paired ratios between every two timed configs; headline = best correct config vs BASE and REF)
#   (v)   exit 0 (2 only if inputs/numpy are missing, checked before any GPU use)
# Overrides (dry runs): NAMES REF BASE NBASE CHECK_TXT MIXED_TXT ORACLE_DIR NRUNS WORK BENCH_BIN_SUFFIX
#   BENCH_SKIP_BUILD BUDGET_S RUN_TIMEOUT SRC_<name>
set -u
T_START=$(date +%s)
CANDS=/workspace/candidates
HERE=$(cd "$(dirname "$0")" && pwd); FINAL=$(cd "$HERE/../.." && pwd)
OUT=${1:-$FINAL/runs/gpu_bench5}
[ -e "$OUT" ] && OUT=${OUT}_$(date +%Y%m%d_%H%M%S)      # never mix files from an earlier run
WORK=${WORK:-/tmp/pfbench5}
CHECK_TXT=${CHECK_TXT:-/workspace/bench/corpus/check.txt}
MIXED_TXT=${MIXED_TXT:-/workspace/bench/corpus/mixed.txt}
ORACLE_DIR=${ORACLE_DIR:-/workspace/experiments/E1/oracle_bf16emu}
NRUNS=${NRUNS:-5}
BUDGET_S=${BUDGET_S:-540}                   # stop starting timing rounds past this (GPU wall budget)
RUN_TIMEOUT=${RUN_TIMEOUT:-150}             # per pf run; a hang drops the config
BIN_NAME=${BENCH_BIN_SUFFIX:-pf_cuda}       # pf_cpu for a CPU dry run
NAMES=${NAMES:-"host_moe_attn_f16 f16_dbuf stack_fd stack_fdm"}
REF=${REF:-host_moe_attn_f16}
BASE=${BASE-initial}                        # "" disables the initial-port runs
NBASE=${NBASE:-3}                           # timed rounds that include BASE (1..NBASE)
declare -A SRC BIN ARGS OK
[ -n "$BASE" ] && SRC[$BASE]=/workspace/implementation/pf_cuda
for n in $NAMES; do SRC[$n]=$CANDS/$n; done
[ -n "${SRC[stack_fdm]:-}" ] && SRC[stack_fdm]=$FINAL/engine
for n in $NAMES $BASE; do v=SRC_$n; [ -n "${!v:-}" ] && SRC[$n]=${!v}; done   # SRC_<name>=dir overrides
CONFIGS=()
for n in $BASE $NAMES; do CONFIGS+=("$n"); BIN[$n]=$n; ARGS[$n]=""; done

[ -n "${CUDA_HOME:-}" ] || . /workspace/env.sh
export PF_MODEL=${PF_MODEL:-/workspace/.cache/pf_model}
PY=python; [ -x /root/.venv/bin/python ] && PY=/root/.venv/bin/python
command -v "$PY" > /dev/null || PY=python3
mkdir -p "$OUT" "$OUT/check" "$OUT/timing" "$WORK"
LOG=$OUT/bench.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for f in "$CHECK_TXT" "$MIXED_TXT" "$ORACLE_DIR/check.jsonl" "$ORACLE_DIR/check.masked.txt" \
         "$ORACLE_DIR/check.lp.gz" "$FINAL/bench/gate/compare.py" "$HERE/gpu_bench5_report.py" "$PF_MODEL"; do
    [ -e "$f" ] || { log "missing input: $f"; exit 2; }
done
for n in $BASE $NAMES; do [ -d "${SRC[$n]}" ] || log "no source dir for $n: ${SRC[$n]} (its build will fail and it is dropped)"; done
case " $NAMES " in *" $REF "*) ;; *) log "REF=$REF not in NAMES"; exit 2;; esac
case " $NAMES " in *" $BASE "*) [ -n "$BASE" ] && { log "BASE=$BASE must not be in NAMES"; exit 2; };; esac
{ nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv 2>&1
  echo "nproc $(nproc)"; lscpu 2>/dev/null | grep -m1 'Model name'; nvcc --version | tail -1; } \
    > "$OUT/env.txt"
# pipelined builds default to (affinity cores - 1) host threads; a container CPU quota can be lower
AFF=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)     # what sched_getaffinity reports
CPUS=$AFF q=max p=1
if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q p < /sys/fs/cgroup/cpu.max
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -gt 0 ] 2> /dev/null || q=max
fi
[ "$q" != max ] && [ $(( (q + p - 1) / p )) -lt "$CPUS" ] && CPUS=$(( (q + p - 1) / p ))
if [ -z "${PF_HOST_THREADS:-}" ] && [ "$CPUS" -lt "$AFF" ]; then
    export PF_HOST_THREADS=$(( CPUS > 1 ? CPUS - 1 : 1 ))
fi
echo "affinity cpus $AFF, quota-aware cpus $CPUS, PF_HOST_THREADS=${PF_HOST_THREADS:-default (affinity - 1)}" >> "$OUT/env.txt"
$PY -I -c 'import numpy' 2> /dev/null || { log "$PY has no numpy (needed by compare.py)"; exit 2; }
for c in "${CONFIGS[@]}"; do echo "$c ${BIN[$c]} ${ARGS[$c]:-(default batch)}"; done > "$OUT/configs.txt"
log "out=$OUT work=$WORK nruns=$NRUNS names=[$NAMES] ref=$REF base=${BASE:-none} (rounds 1..$NBASE) budget=${BUDGET_S}s"

# ---------------------------------------------------------------- (i) build
build_one() {   # name -> $WORK/<name>/<BIN_NAME>; sources sha12 -> $WORK/<name>/SHA12
    local n=$1 d=$WORK/$1
    rm -rf "$d" && mkdir -p "$d"
    (cd "${SRC[$n]}" && cp -- build.sh $(ls *.c *.cu *.h *.inc 2>/dev/null) "$d"/)
    (cd "$d" && cat build.sh $(ls *.c *.cu *.h *.inc | LC_ALL=C sort) | sha256sum | cut -c1-12 > SHA12)
    if [ -n "${BENCH_SKIP_BUILD:-}" ]; then        # dry run: CPU binary only
        (cd "$d" && gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cpu pf.c -lm)
    else
        sh "$d/build.sh"
    fi
}
for n in $BASE $NAMES; do
    build_one "$n" > "$OUT/build_$n.log" 2>&1 &
    eval "BPID_$n=$!"
done
declare -A BUILT
for n in $BASE $NAMES; do
    eval "p=\$BPID_$n"
    if wait "$p" && [ -x "$WORK/$n/$BIN_NAME" ]; then BUILT[$n]=1; log "build $n ok ($(cat $WORK/$n/SHA12))"
    else BUILT[$n]=0; log "build $n FAILED (see build_$n.log)"; fi
done
for c in "${CONFIGS[@]}"; do OK[$c]=${BUILT[${BIN[$c]}]}; done

# ---------------------------------------------------------------- (ii) check.txt runs + gate + identity
memwatch() {   # sample GPU memory.used (MiB) every 200 ms into $1 until killed
    while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 >> "$1"; sleep 0.2; done
}
CMP_PIDS=()
for c in "${CONFIGS[@]}"; do
    [ "${OK[$c]}" = 1 ] || continue
    b=$WORK/${BIN[$c]}/$BIN_NAME f=$OUT/check/$c a=${ARGS[$c]}
    memwatch "$f.gpumem" & MW=$!
    timeout "$RUN_TIMEOUT" "$b" --lines -j $a -f "$CHECK_TXT" > "$f.jsonl" 2> "$f.jsonl.err"; e1=$?
    kill "$MW" 2> /dev/null; wait "$MW" 2> /dev/null
    if [ "$e1" = 0 ] &&
       timeout "$RUN_TIMEOUT" "$b" --lines $a -f "$CHECK_TXT" > "$f.masked.txt" 2> "$f.masked.err" &&
       timeout "$RUN_TIMEOUT" "$b" --lines --dump-logprobs $a -f "$CHECK_TXT" 2> "$f.lp.err" | gzip -1 > "$f.lp.gz" &&
       [ "${PIPESTATUS[0]}" = 0 ]; then
        log "check runs $c ok (peak GPU memory.used $(sort -n "$f.gpumem" 2> /dev/null | tail -1) MiB); compare started (CPU, background)"
        $PY -I "$FINAL/bench/gate/compare.py" --min-agree 98.0 \
            --ref-jsonl "$ORACLE_DIR/check.jsonl" --cand-jsonl "$f.jsonl" \
            --ref-masked "$ORACLE_DIR/check.masked.txt" --cand-masked "$f.masked.txt" \
            --ref-lp "$ORACLE_DIR/check.lp.gz" --cand-lp "$f.lp.gz" \
            --out "$f.vs_bf16emu.json" > "$f.vs_bf16emu.txt" 2>&1 &
        CMP_PIDS+=($!)
    else
        OK[$c]=0; log "check runs $c FAILED (see $f.*.err)"
    fi
done
for c in "${CONFIGS[@]}"; do echo "$c ${OK[$c]}"; done > "$OUT/check/ok.txt"   # 1 = built and check runs ok
r=$OUT/check/$REF
for c in "${CONFIGS[@]}"; do
    [ "$c" = "$REF" ] && continue
    if [ "${OK[$c]}" != 1 ] || [ "${OK[$REF]}" != 1 ]; then echo "$c n/a n/a n/a"; continue; fi
    j=$(cmp -s "$r.jsonl" "$OUT/check/$c.jsonl" && echo same || echo DIFF)
    m=$(cmp -s "$r.masked.txt" "$OUT/check/$c.masked.txt" && echo same || echo DIFF)
    l=$(cmp -s <(gzip -dc "$r.lp.gz") <(gzip -dc "$OUT/check/$c.lp.gz") && echo same || echo DIFF)
    echo "$c $j $m $l"
done > "$OUT/check/identity.txt"           # config jsonl masked logprob-dump (vs REF; BASE informational)
while read -r c j m l; do log "identity vs $REF: $c jsonl=$j masked=$m lp=$l"; done < "$OUT/check/identity.txt"
for p in "${CMP_PIDS[@]}"; do wait "$p"; done
$PY -I "$HERE/gpu_bench5_report.py" --gate "$OUT" "${CONFIGS[@]}" > "$OUT/check/gate.txt" 2> "$OUT/check/gate.err"
while read -r c verdict rest; do log "gate $c: $verdict $rest"; done < "$OUT/check/gate.txt"
verdict() { awk -v n="$1" '$1 == n { print $2 }' "$OUT/check/gate.txt"; }

# ---------------------------------------------------------------- mixed.txt run helper
run_mixed() {   # config tag -> 0 if ok; stats in OUT/timing/<config>.<tag>.err, md5 in .md5
    local c=$1 t=$2 e=$OUT/timing/$1.$2.err
    if timeout "$RUN_TIMEOUT" "$WORK/${BIN[$c]}/$BIN_NAME" --stats --lines ${ARGS[$c]} -f "$MIXED_TXT" \
            > "$WORK/out_$c.txt" 2> "$e"; then
        md5sum "$WORK/out_$c.txt" | cut -c1-32 > "$OUT/timing/$c.$t.md5"
        log "$t $c: $(grep -E '\| e2e [0-9.]+ms' "$e" | grep -oE 'e2e [0-9.]+ms \([0-9.]+ tok/s\)' | head -1)"
        return 0
    fi
    echo FAILED > "$OUT/timing/$c.$t.md5"; log "$t $c: run FAILED (see $e)"
    return 1
}
e2e_of() { grep -oE '\| e2e [0-9.]+ms \([0-9.]+ tok/s\)' "$1" 2> /dev/null | head -1 | grep -oE '[0-9.]+ tok/s' | cut -d' ' -f1; }

# ---------------------------------------------------------------- (iii) mixed.txt timing
TIMED=() WHY=()
[ "${OK[$REF]}" = 1 ] && { TIMED+=("$REF"); WHY+=("$REF reference ($(verdict "$REF"))"); }
for n in $NAMES; do
    [ "$n" = "$REF" ] && continue
    if [ "$(verdict "$n")" = PASS ]; then TIMED+=("$n"); WHY+=("$n gate PASS")
    else WHY+=("$n not timed: gate $(verdict "$n")"); fi
done
TB=0
if [ -n "$BASE" ]; then
    if [ "$(verdict "$BASE")" = PASS ]; then TB=1; WHY+=("$BASE gate PASS, timed in rounds 1..$NBASE")
    else WHY+=("$BASE not timed: gate $(verdict "$BASE")"); fi
fi
printf '%s\n' "${WHY[@]}" > "$OUT/timing/timed.txt"
log "timing on $(basename "$MIXED_TXT"): ${TIMED[*]} (round 0 = warmup; order rotated per round)$([ "$TB" = 1 ] && echo " + $BASE in rounds 1..$NBASE")"
ROUNDS_DONE=-1 T_ROUND=0
for r in $(seq 0 "$NRUNS"); do
    now=$(date +%s)
    if [ "$r" -gt 1 ] && [ $(( now - T_START + T_ROUND )) -gt "$BUDGET_S" ]; then
        log "time guard: round $r would end past ${BUDGET_S}s (round takes ~${T_ROUND}s); stopping"
        break
    fi
    k=${#TIMED[@]}
    [ "$k" -gt 0 ] || break
    order=()
    for j in $(seq 0 $((k - 1))); do order+=("${TIMED[$(( (r + j) % k ))]}"); done
    if [ "$TB" = 1 ] && [ "$r" -ge 1 ] && [ "$r" -le "$NBASE" ]; then   # BASE at slot (r-1) mod (k+1)
        p=$(( (r - 1) % (k + 1) ))
        order=("${order[@]:0:$p}" "$BASE" "${order[@]:$p}")
    fi
    echo "r$r ${order[*]}" >> "$OUT/timing/order.txt"
    for c in "${order[@]}"; do
        if ! run_mixed "$c" "r$r"; then
            log "$c dropped from later rounds"
            if [ "$c" = "$BASE" ]; then TB=0; continue; fi
            t2=(); for x in "${TIMED[@]}"; do [ "$x" != "$c" ] && t2+=("$x"); done; TIMED=("${t2[@]}")
        fi
    done
    ROUNDS_DONE=$r
    T_ROUND=$(( $(date +%s) - now ))
done
echo "$ROUNDS_DONE" > "$OUT/timing/rounds_done.txt"
rm -f "$WORK"/out_*.txt
T_GPU=$(date +%s)
log "GPU work done in $((T_GPU - T_START)) s (timed rounds completed: $ROUNDS_DONE of $NRUNS)"

# ---------------------------------------------------------------- (iv) report
$PY -I "$HERE/gpu_bench5_report.py" "$OUT" "$WORK" "$NRUNS" "$NBASE" "$((T_GPU - T_START))" "$REF" "${BASE:--}" \
    "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"
log "total $(( $(date +%s) - T_START )) s"
exit 0
