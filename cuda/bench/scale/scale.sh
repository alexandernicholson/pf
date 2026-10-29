#!/usr/bin/env bash
# Multi-GPU data-parallel scaling of a pf_cuda build on one box.
#   bash final/bench/scale/scale.sh OUT [N_LIST]   (default N_LIST="1 2 4", capped at #GPUs)
# (final/ copy of candidates/scale.sh: default SRC is final/engine; never run on >1 GPU in this project)
# Env: SRC (candidate dir, default final/engine), CORPUS (default bench/corpus/mixed.txt),
#      BIN (prebuilt binary; skips the build), NRUNS (timed runs per N, default 3),
#      TOKENS (corpus o200k tokens, default 7371164 = mixed.txt), NG (override GPU count).
# One process per GPU (CUDA_VISIBLE_DEVICES=i), each on a contiguous line-chunk of the corpus.
#   agg_e2e  = TOKENS / max over processes of its own --stats e2e (load excluded, like the headline)
#   agg_wall = TOKENS / wall time from launch to last exit (includes process start + model load)
# The merged output is compared with the N=1 output (md5 + per-doc agreement).
set -uo pipefail
OUT=${1:?usage: scale.sh OUT [N_LIST]}; NLIST=${2:-"1 2 4"}
W=/workspace; [ -n "${CUDA_HOME:-}" ] || . $W/env.sh
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$W/final/engine}; CORPUS=${CORPUS:-$W/bench/corpus/mixed.txt}; NRUNS=${NRUNS:-3}
export TOKENS=${TOKENS:-7371164}
mkdir -p "$OUT"; exec > >(tee -a "$OUT/scale.log") 2>&1
NG=${NG:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')}; NC=$(nproc)
echo "gpus=$NG cores=$NC src=$SRC corpus=$CORPUS"; nvidia-smi -L 2>/dev/null || true
if [ -z "${BIN:-}" ]; then
  B=/tmp/pfscale/build; rm -rf $B; mkdir -p $B; cp -r "$SRC"/. $B/
  ( cd $B && ./build.sh > "$OUT/build.log" 2>&1 ) || { echo "BUILD FAILED"; tail -20 "$OUT/build.log"; exit 1; }
  BIN=$B/pf_cuda
fi
echo "docs=$(wc -l < "$CORPUS") bin=$BIN"
: > "$OUT/runs.txt"

run_n() {  # $1=N $2=tag ; prints "wall_ms rc"
  local N=$1 tag=$2 d="$OUT/n$1"; mkdir -p "$d/chunks"
  if [ ! -f "$d/chunks/.done" ]; then
    if [ "$N" = 1 ]; then cp "$CORPUS" "$d/chunks/c00"; else split -n l/$N -d -a 2 "$CORPUS" "$d/chunks/c"; fi
    touch "$d/chunks/.done"
  fi
  local th=$(( NC / N - 1 )); [ $th -lt 1 ] && th=1
  local t0 t1 i c rc=0; local -a pids=()
  t0=$(date +%s%N)
  for ((i=0;i<N;i++)); do
    c=$(printf "%02d" $i)
    CUDA_VISIBLE_DEVICES=$i PF_HOST_THREADS=$th timeout 300 "$BIN" --stats --lines -f "$d/chunks/c$c" \
        > "/tmp/pfscale_out_${N}_${i}.txt" 2> "$d/$tag.p$i.err" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  t1=$(date +%s%N)
  for ((i=0;i<N;i++)); do cat "/tmp/pfscale_out_${N}_${i}.txt"; done > "$d/$tag.out.txt"
  echo "$(( (t1-t0)/1000000 )) $rc"
}

for N in $NLIST; do
  if [ "$N" -gt "$NG" ]; then echo "skip N=$N (only $NG GPUs)"; continue; fi
  for r in $(seq 0 $NRUNS); do
    tag=r$r; read -r wall rc < <(run_n $N $tag)
    echo "N=$N $tag wall_ms=$wall rc=$rc md5=$(md5sum < "$OUT/n$N/$tag.out.txt" | cut -c1-12)"
    echo "$N $tag $wall $rc" >> "$OUT/runs.txt"
  done
done

python3 -I "$HERE/scale_report.py" "$OUT"
echo "scale.sh done"
