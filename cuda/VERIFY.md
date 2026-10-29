# VERIFY: CPU-box verification of final/ (E29, 2026-10-28)

**Box.** CPU only, no GPU, nvcc `cuda_12.8.r12.8/compiler.35583870_0` from `/workspace/env.sh`.

**Reproduce.** `bash /workspace/experiments/E29/verify_all.sh`. The full log is in `experiments/E29/logs/verify_all.log`.

**Two limits on this verification.**
- **No GPU numerics.** Nothing here runs the GPU forward. The GPU numbers and gate results in `README.md` and `REPORT.md` come from the cited H100 runs. This file shows that `final/` contains exactly the code those runs measured, and that it builds and runs its host paths correctly.
- **No binaries shipped.** `final/` holds sources only. The script removes its build outputs at the end, because the host objects use `-march=native`.

| # | check | result | evidence |
|---|---|---|---|
| 1 | **Source identity.** Every non-README file in `engine/`, `server/` and `extras/fp8/` is byte-identical to `candidates/stack_fdm`, `candidates/serve2` and `candidates/f16_fp8`. | **yes**, all three | `experiments/E29/logs/verify_all.log` |
| 2 | **Source ids match the GPU-measured candidates.** The ids use the same recipe as `gpu_bench5.sh` and `serve_bench2.sh`. | engine `17370154d25f` = `stack_fdm@17370154d25f` (E23, E25); server `d57e388c49dc` = `serve2@d57e388c49dc` (E26); fp8 `d12bbc379099` = `f16_fp8@d12bbc379099` (E23 fp8) | same |
| 3 | **Builds** with each folder's own `build.sh`, in place. | `engine/build.sh` exit 0; `PF_BATCH_INVARIANT=1 server/build.sh` exit 0; `server/build.sh` exit 0; `extras/fp8/build.sh` exit 0. Each produces `pf_cuda` and `pf_cpu`. | `experiments/E29/logs/build_*.log`, `build_status.txt` |
| 4 | **Build-mode sanity.** | BI server binary has 0 `cublasGemmEx` references (cuBLAS GEMMs replaced); the default server, engine and fp8 binaries have 1. Engine MoE ptxas: all 8 entries 0 stack, 0 spill bytes. | `experiments/E29/logs/build_status.txt`, `engine_moe_spills.txt` |
| 5 | **sm_90a SASS identity.** `final/engine/pf_cuda` vs `candidates/stack_fdm/pf_cuda` (whitespace-normalized `cuobjdump -sass`). | **identical**, 18 kernels | `experiments/E29/sass/` |
| 6 | **Replay identity** (`bench/gate/replay_check.sh`, `PF_NULL_FORWARD=replay:` with the fixed CPU oracle's log-probs on check.txt). Covers engine, default server, BI server and fp8, each at batch default/4096/1000 × `PF_HOST_THREADS` default/3. | **24/24 byte-identical** to the oracle (`-j` jsonl and masked text); 1534 docs, 353,505 tokens, 3 or 94 batches | `experiments/E29/replay_check/results.txt` |
| 7a | **Server smoke, real CPU forward.** `server/pf_cpu --serve` with the README's curl examples. | `/health`, `/redact` (`text` and `texts`), `/redact/openai`, `/redact/anthropic` and `/stats` all answer with correct masks. The `/redact` `texts` response equals `pf_cpu --lines` on the same docs. | `experiments/E29/server_smoke/results.txt` |
| 7b | **Server smoke, CUDA-linked binary.** `server/pf_cuda --serve` with `PF_NULL_FORWARD=1`. | Starts, listens and answers every route. Masks are synthetic by design. | same |
| 8 | **Bench script dry runs** (`pf_cpu`, small inputs; checks path wiring only). | All exit 0. | `experiments/E29/dry/` |
| 9 | **Gate script.** `bench/gate/gate.sh` on `pf_cpu` with E7's 40-doc check slice. | Fp32 mini oracle: PASS, 100% / 0.000. Bf16-emu mini oracle: **FAIL**, exit 1, mean \|dlp\| 0.263 > 0.23 (expected: the fp32 CPU forward vs the bf16-emu oracle, as in E7/E13). So the gate can fail. | `experiments/E29/dry/gate/` |

Details for row 8:
- `bench/batch/gpu_bench5.sh` (two runs) builds `stack_fdm` from `final/engine`, with sources id `17370154d25f`. With `BASE=` (empty) the initial port is now really skipped.
- `bench/serve/serve_bench2.sh` (`PF_NULL_FORWARD=1`):
  - builds both server modes from `final/server`;
  - the invariance test, E2E and full server matrix all run, with 100% response agreement, 0 errors and exact500 500/500 before and after;
  - gemm_bench is skipped without a GPU;
  - the gate fails by design on synthetic log-probs.
- `bench/scale/scale.sh` with a faked `NG=2`: N=1 and N=2 merged outputs have the same md5.

## Not verified here
- **Everything that needs an H100.** That includes speed, the gate on GPU numerics and server latency. These were measured in E23, E25 and E26, on exactly these sources (rows 1, 2 and 5).
- **The A100 path** (`candidates/portable`, in progress) and **B200** (`candidates/b200`, never run).
- **Multi-GPU scaling.**
