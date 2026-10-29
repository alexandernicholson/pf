# final/bench: gate, identity and timing scripts

Every script can be run from any working directory. Each one finds its siblings through `$(dirname "$0")` and the build trees through `final/`. The data inputs are absolute `/workspace` paths, and each can be overridden by the env var named in the script's header:

| input | default path |
|---|---|
| corpora | `/workspace/bench/corpus/mixed.txt`, `check.txt` |
| CPU oracle | `/workspace/experiments/E1/oracle_bf16emu/` |
| model | `$PF_MODEL` = `/workspace/.cache/pf_model` |
| replay inputs | `/workspace/experiments/E2/cpu/` |

| script | what | copied from | changes in this copy |
|---|---|---|---|
| `gate/compare.py` | the correctness comparison: span, masked and log-prob agreement vs the oracle | `bench/compare.py` | none |
| `gate/gate.sh BIN OUT [pf args]` | check.txt runs + compare.py + the gate limits (≥98.0% span and masked agreement, ≤0.23 mean \|dlp\|); exit 1 on FAIL | new (same steps as gpu_bench5.sh (ii)) | |
| `gate/replay_check.sh OUT NAME=BIN ...` | CPU-only host-glue identity: oracle log-probs replayed through each binary must give byte-identical `-j` and masked output at batch default/4096/1000 × threads default/3 | new (generalizes `experiments/E21/replay_check/run.sh`) | |
| `batch/gpu_bench5.sh [OUT]` + `gpu_bench5_report.py` | the H100 batch benchmark behind the headline: builds, gate, identity vs REF, rotated paired timing, `summary.json` | `candidates/gpu_bench5.sh`, `candidates/gpu_bench5_report.py` | `stack_fdm` builds from `final/engine`; compare.py and the report come from `final/bench`; default OUT is `final/runs/gpu_bench5`; `BASE=` (empty) now really disables the initial-port runs (the original `${BASE:-initial}` ignored an empty value). REF, f16_dbuf and stack_fd still build from `/workspace/candidates`, and the initial port from `/workspace/implementation/pf_cuda`; they are needed only for the paired ratios. |
| `serve/serve_bench2.sh [OUT]` + `bi_invariance.py`, `bi_report.py`, `loadgen.py`, `gemm_bench/gemm_bench.cu` | the H100 server benchmark: both server builds, gemm_bi microbench, batch-invariance test, gate, batch E2E, load matrix, `results.json` | `experiments/E22/` (+ `experiments/E11/loadgen.py`) | sources from `final/server`; helpers from this folder; default OUT is `final/runs/serve` |
| `scale/scale.sh OUT [N_LIST]` + `scale_report.py` | data-parallel multi-GPU scaling, one process per GPU on line-chunks, merged output checked vs N=1 | `candidates/scale.sh`, `candidates/scale_report.py` | default SRC is `final/engine`; never run on more than 1 GPU |

**Typical H100 session:**
```sh
bash /workspace/setup.sh && source /workspace/env.sh
bash /workspace/final/bench/batch/gpu_bench5.sh /workspace/final/runs/batch     # ~4 min; headline + paired ratios
NAMES=stack_fdm REF=stack_fdm BASE= bash /workspace/final/bench/batch/gpu_bench5.sh /workspace/final/runs/engine_only
bash /workspace/final/bench/serve/serve_bench2.sh /workspace/final/runs/serve   # ~6.5 min
bash /workspace/final/bench/gate/gate.sh /workspace/final/engine/pf_cuda /workspace/final/runs/gate
```

Each batch and server script was dry-run on a CPU box with `pf_cpu` and small inputs (`experiments/E29/dry/`, see `final/VERIFY.md`). Those dry runs checked the wiring only: the timings and gate verdicts are meaningless there.
