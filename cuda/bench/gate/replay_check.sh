#!/bin/bash
# Host-glue identity check (as experiments/E21/replay_check): each pf_cuda replays the fixed CPU oracle's
# log-probs via PF_NULL_FORWARD=replay: and must give -j and masked output byte-identical to the oracle.
# No GPU needed (the replay path skips gpu_init). Small --batch values force many batches, so the
# double-buffered pipeline alternates slots; PF_HOST_THREADS=3 vs default checks the threaded host pipeline.
#   bash final/bench/gate/replay_check.sh OUT_DIR NAME=/path/to/pf_cuda [NAME=/path/to/pf_cuda ...]
# Writes OUT_DIR/results.txt; exits 1 if any run differs.
set -u
OUT=${1:?usage: replay_check.sh OUT_DIR NAME=BIN ...}; shift
. /workspace/env.sh
mkdir -p "$OUT"
C=/workspace/bench/corpus/check.txt
RJ=/workspace/experiments/E2/cpu/check.run1.jsonl RM=/workspace/experiments/E2/cpu/check.fixed.masked.txt
LP=/tmp/final_replay_lp.$$.f32
python3 -I /workspace/experiments/E4/tools/lp_to_f32.py /workspace/experiments/E2/cpu/check.fixed.logprobs.txt.gz $LP > /dev/null
: > "$OUT/results.txt"; fail=0
for nb in "$@"; do
    n=${nb%%=*}; bin=${nb#*=}
    for bt in "" 4096 1000; do
        for th in "" 3; do
            tag=$OUT/$n.b${bt:-def}.t${th:-def}
            env -u PF_HOST_THREADS ${th:+PF_HOST_THREADS=$th} PF_NULL_FORWARD=replay:$LP \
                "$bin" --stats --lines -j ${bt:+--batch $bt} -f $C > $tag.jsonl 2> $tag.err
            env -u PF_HOST_THREADS ${th:+PF_HOST_THREADS=$th} PF_NULL_FORWARD=replay:$LP \
                "$bin" --lines ${bt:+--batch $bt} -f $C > $tag.masked.txt 2>> $tag.err
            j=DIFF; cmp -s $tag.jsonl $RJ && j=identical
            m=DIFF; cmp -s $tag.masked.txt $RM && m=identical
            [ $j = identical ] && [ $m = identical ] || fail=1
            echo "replay $n batch=${bt:-default} threads=${th:-default}: jsonl $j, masked $m; $(grep -o '[0-9]* docs, [0-9]* tok, [0-9]* batches' $tag.err | head -1)" | tee -a "$OUT/results.txt"
            rm -f $tag.jsonl $tag.masked.txt
        done
    done
done
rm -f $LP
echo "overall: $([ $fail = 0 ] && echo PASS || echo FAIL)" | tee -a "$OUT/results.txt"
exit $fail
