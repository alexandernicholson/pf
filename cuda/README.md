# pf-cuda: OpenAI privacy-filter PII redaction on NVIDIA GPUs

This folder is the final deliverable of the pf CUDA port. [pf](https://github.com/alexandernicholson/pf) is a one-file C program that runs OpenAI's `privacy-filter` model to redact PII. Here its forward pass runs on the GPU and its host side is parallel, so pf keeps its exact CLI and output formats and gains an HTTP server.

**Measured result (1x H100 80GB HBM3).** On mixed.txt (47,928 docs, 7,371,164 tokens) the engine reaches **2,650,119 tok/s end to end**, median of 5 runs (`experiments/E23/gpu_run5/summary.json`). That is:
- 5.45x the initial CUDA port;
- 277x the reference PyTorch+Triton GPU implementation (9,555 tok/s);
- about 485x CPU pf (~5,470 tok/s, 8 threads).

A fresh H100 box reproduced it at 2,648,663 tok/s (`experiments/E25/results.json`). The engine passes the correctness gate described below. On A100-80GB, the portable build reaches 1,122,849 tok/s and passes the gate (see `PLATFORMS.md`).

| folder | what | source (byte-identical copy) | GPU-verified id |
|---|---|---|---|
| `engine/` | best batch build: parallel host pipeline, double-buffered forward, fused qkv→fp16 epilogue, fp16 tensor-core attention, warp-specialized wgmma+TMA fused MoE | `candidates/stack_fdm` | `stack_fdm@17370154d25f` (E23, E25) |
| `server/` | pf `--serve` with adaptive dynamic batching and keep-alive, plus an optional batch-invariant build | `candidates/serve2` | `serve2@d57e388c49dc` (E26) |
| `extras/fp8/` | opt-in FP8 (e4m3) expert GEMMs, `PF_FP8_EXPERTS=1`; **fails the gate**, so it is off by default | `candidates/f16_fp8` | `f16_fp8@d12bbc379099` (E23 gpu_run_fp8) |
| `bench/` | gate, replay check, batch timing, server load test and multi-GPU scaling scripts, adjusted to run from `final/` | `bench/compare.py`, `candidates/gpu_bench5.sh`, `experiments/E22/serve_bench2.sh`, `candidates/scale.sh` (list and changes in `bench/README.md`) | |
| `PLATFORMS.md` | H100 and A100 results, plus B200 and multi-GPU status | | |
| `REPORT.md` | full technical report: method, bound, every optimization with paired ratios, kernels, precision and FP8 studies, server | | |
| `VERIFY.md` | CPU-box verification of this folder: builds, SASS identity, replay identity, server smoke test, script dry runs | | |

Copies are of the sources only; each folder's README has a status note added at the top. The ids are sha256 prefixes of the sources, computed the same way the GPU scripts compute them, and they match the ids in the cited GPU runs (`VERIFY.md`). So the code in this folder is exactly the code that was measured.

## Requirements
- **GPU: NVIDIA H100 (sm_90a).**
  - The engine's MoE uses wgmma, TMA and setmaxnreg, and all published headline numbers are from 1x H100 80GB HBM3.
  - Peak device memory on check.txt was 9.5 GB (9,539 MiB).
  - For A100 (sm_80) use the portable fat binary `candidates/portable` (see `PLATFORMS.md`).
- **CUDA 12.8 toolkit.** It provides nvcc and cuBLAS; `/workspace/setup.sh` installs it into `/workspace/.cache/cuda-12.8`, and `source /workspace/env.sh` sets `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH` and `PF_MODEL`. Any CUDA 12.x with sm_90a support should work, but only 12.8 was tested. The driver used was 610.57.04.
- **Model.** `/workspace/setup.sh` downloads it into `/workspace/.cache/pf_model`. The directory holds `model.safetensors` and `config.json` from `openai/privacy-filter` (`original/`), plus `o200k_base.tiktoken`. Point `PF_MODEL` or `--model DIR` at it.
- **Host.** gcc, x86-64 (built with `-march=native`, so build on the machine that runs it) and Linux/glibc. The host pipeline uses all cores but one by default; the H100 runs had 8 cores in `nproc` and 24 in the affinity set.

## Build
```sh
source /workspace/env.sh                         # CUDA 12.8 toolkit + PF_MODEL (build.sh sources it itself if CUDA_HOME is unset)
sh final/engine/build.sh                         # -> final/engine/pf_cuda (GPU) and pf_cpu (same pf.c, CPU forward)
sh final/server/build.sh                         # -> final/server/pf_cuda, default server build (cuBLAS dense GEMMs)
PF_BATCH_INVARIANT=1 sh final/server/build.sh    # -> final/server/pf_cuda, batch-invariant build (same output path)
sh final/extras/fp8/build.sh                     # -> final/extras/fp8/pf_cuda, opt-in FP8 experts
sh /workspace/candidates/portable/build.sh       # fat binary for sm_80 / sm_90a / sm_100 (A100: run with PF_MOE_CFG=4)
```

- Each build takes under a minute and writes into its own folder. `ARCH=` overrides the default `sm_90a`, but the engine needs sm_90a.
- `engine/build.sh` also writes `moe_ptxas.log`, which shows 0 spill bytes in all 8 MoE entries.
- **Build on the machine you run on.** The host objects use `-march=native`, so `final/` ships sources only, with no prebuilt binaries.

## Batch CLI (`engine/pf_cuda`; the same flags work on `server/pf_cuda` and `pf_cpu`)
```sh
export PF_MODEL=/workspace/.cache/pf_model
final/engine/pf_cuda "Call Alice Smith at 415-555-0199"        # masked text on stdout
final/engine/pf_cuda -j "Email jane.roe@example.org"           # JSON spans with byte offsets
final/engine/pf_cuda --lines -f docs.txt > masked.txt          # batch mode: one doc per line in, one line out
final/engine/pf_cuda --lines -j -f docs.txt > spans.jsonl      # batch mode, JSONL spans
final/engine/pf_cuda --stats --lines -f docs.txt > /dev/null   # timing breakdown on stderr (tok/s, GPU phase split)
final/engine/pf_cuda --lines --batch 65536 -f docs.txt         # token budget per packed GPU batch (default 131072 on GPU)
PF_HOST_THREADS=8 final/engine/pf_cuda --lines -f docs.txt     # host worker threads (default: usable cores - 1)
```

Example output, from the CPU build of the same `pf.c`:

```
$ pf_cpu -j "Email jane.roe@example.org or call 020 7946 0958."
{"spans":[{"label":"private_email","start":6,"end":26,"text":"jane.roe@example.org"},{"label":"private_phone","start":35,"end":48,"text":"020 7946 0958"}]}
```

Other flags are as in upstream pf:
- `-f -` reads stdin;
- `--ctx N` sets the context window (default 4096);
- `--argmax` uses per-token argmax instead of Viterbi;
- `--dump-logprobs` writes the per-token log-probs, which the gate uses;
- `--model DIR` sets the model directory.

`--stats` prints, for example (an `engine/` run on mixed.txt from `experiments/E23/gpu_run5/timing/`):
```
pipeline: 23 threads, 689 blocks | worker-time tokenize 1541.7ms viterbi 608.2ms ... | main pack 16.0ms forward 2520.3ms scatter 207.5ms ...
gpu phases (ms, ...): h2d/d2h 32.3 | embed 18.1 | moe-reduce+norm 479.9 | qkv gemm 222.6 | bias+rope 147.0 | attention 285.6 | ... | moe fused (route+gemm1+swiglu+gemm2) 1347.9 | head 47.9
pipelined | tokenizer load 21ms | model load+init 752ms | forward 2520ms (2924674.9 tok/s) | e2e 2794ms (2638296.0 tok/s)
```

"e2e" covers tokenize + forward + decode + output and excludes process start, file read and model load (about 0.75 s). That is the headline metric.

## HTTP server (`server/pf_cuda --serve PORT`)
```sh
final/server/pf_cuda --serve 8080            # listens on 127.0.0.1:8080 once the engine is up
curl -s localhost:8080/health                # {"status":"ok"}
curl -s localhost:8080/redact -d '{"text":"Call Alice Smith at 415-555-0199 or alice.smith@example.com."}'
# {"text":"Call <PRIVATE_PERSON> at <PRIVATE_PHONE> or <PRIVATE_EMAIL>."}
curl -s localhost:8080/redact -d '{"texts":["My name is John Doe and I live at 12 Baker Street, London.","Card 4111 1111 1111 1111, exp 09/27."]}'
# {"texts":["My name is <PRIVATE_PERSON> and I live at <PRIVATE_ADDRESS>.","Card <ACCOUNT_NUMBER>, exp <PRIVATE_DATE>."]}
curl -s localhost:8080/redact/openai -d '{"model":"gpt-x","messages":[{"role":"user","content":"I am Maria Garcia, email maria@corp.example"}]}'
# {"model":"gpt-x","messages":[{"role":"user","content":"I am <PRIVATE_PERSON>, email <PRIVATE_EMAIL>"}]}
curl -s localhost:8080/redact/anthropic -d '{"model":"claude-x","system":"You help Bob Lee.","messages":[{"role":"user","content":"my api key is sk-live-99fKq2"}]}'
# {"model":"claude-x","system":"You help <PRIVATE_PERSON>.","messages":[{"role":"user","content":"my api key is <SECRET>"}]}
curl -s localhost:8080/stats                 # cumulative batcher counters: forwards, jobs, docs, tokens, run_docs_ms, linger_ms
```

The example responses above are real outputs of this `pf.c` with the CPU forward, recorded in `experiments/E29/server_smoke/results.txt`. The GPU build serves the same routes.

**Routes.** The response is the request JSON, unchanged except for the masked string values:

| route | masks |
|---|---|
| `/redact` | `text`, `texts[*]` |
| `/redact/openai` | `messages[*].content` (string or `parts[].text`) |
| `/redact/anthropic` | `system`, `system[*].text`, `messages[*].content` |

**Flags.**

| flag | default | meaning |
|---|---|---|
| `--linger MS` | 8 | Maximum wait for more work when the engine is idle. The adaptive policy dispatches immediately when jobs queued during the previous forward, when queued tokens reach `--dispatch-tok`, when no other connection is still being read, or when no job arrived for `--linger-gap` µs. |
| `--linger-gap US` | 1000 | Idle gap that ends the linger early. |
| `--dispatch-tok N` | `--batch` (131072) | Queued-token threshold for immediate dispatch. |
| `--linger-fixed` | off | Upstream policy: always sleep `--linger` ms. It gave 90 docs/s at C=1 vs 431 adaptive (`experiments/E14`). |
| `--no-keepalive` | keep-alive on | Close after every response, as upstream. Keep-alive is HTTP/1.1's default; HTTP/1.0 clients need `Connection: keep-alive`. |
| `--batch N` | 131072 | Token budget per forward. |

- All CUDA work runs on one batcher thread; connection threads tokenize and decode.
- **Batch-invariant build** (`PF_BATCH_INVARIANT=1 sh server/build.sh`):
  - A document's log-probs are bit-identical whatever else shares its batch, so a server response equals the batch-mode output byte for byte. In E26, all 488,091 server responses were identical to batch mode, against 99.12% for the default build.
  - It replaces cuBLAS with a fixed-tile GEMM (`gemm_bi.cu`) and aligns the attention key steps (`-DATTN_BI`).
  - Cost: 0.91x of the default build's batch E2E. It announces itself on stderr at start-up.

Measured server throughput (H100, E26):

| load | default build | batch-invariant build |
|---|---|---|
| 1 client, short docs | 461 docs/s, p50 2.16 ms | 417 docs/s, p50 2.40 ms |
| 64 clients | 5,704 docs/s, p50 10.9 ms | 7,353 docs/s, p50 7.9 ms |
| 256 clients | 11,614 docs/s (1.79M tok/s) | 11,647 docs/s (1.79M tok/s) |
| 16 clients × 64-doc requests | 2.45M tok/s | 2.15M tok/s |
| responses identical to batch mode | 99.12% | 100% |

## Flags and environment
| variable / flag | build | effect |
|---|---|---|
| `PF_MODEL=DIR` | all | Model directory (default `./model`). |
| `PF_HOST_THREADS=N` | all | Host worker threads for tokenize/Viterbi/format (default: usable cores − 1 on GPU builds). |
| `PF_BATCH_INVARIANT=1` (build time) | server | Batch-invariant engine, see above. Passes the gate at 98.57% / 0.208. |
| `PF_FP8_EXPERTS=1` (run time) | `extras/fp8` | **Opt-in approximation.** e4m3 expert GEMMs with per-column weight scales and per-token activation scales. +9.0% E2E, but 95.50% doc agreement and 0.525 mean \|dlp\|, which **fails the gate**. `PF_FP8_CFG=g1,g2` selects the tiles (default `160s3p0,160s3p0`). Unset, the build is byte-identical to the fp16-attention reference build. |
| `PF_MOE_CFG=N` (run time) | `candidates/portable` | Portable MoE tile config. Use 4 on A100 (1.20x over 0, identical output). |
| `PF_MOE_PORTABLE=1` (run time) | `candidates/portable`, `candidates/b200` | Forces the portable mma.sync MoE on H100 (testing the A100/B200 path). |
| `PF_NULL_FORWARD=1` / `=replay:FILE` | all | Test hooks. `1` gives synthetic log-probs with no GPU; `replay:` replays saved fp32 log-probs. Used by `bench/gate/replay_check.sh`. |
| `PF_ATTN_SIMT=1` | engine, server (default build) | Debug only: the old fp32 SIMT attention. |
| `PF_GPU_SKIP_LAYER=k` | engine, server | Debug only: skips block k. E1's deliberately wrong run (block 3 skipped, `experiments/E1/wrong_skip3`) scored 45.31% and was rejected by the gate. |

## Correctness method
- **Gate.** check.txt (1,534 docs, 353,505 tokens, including long multi-window docs and edge cases) is compared with a CPU oracle: upstream pf with bf16 rounding of matmul inputs (`experiments/E1/oracle_bf16emu`). `bench/gate/compare.py --min-agree 98.0` checks per-document span agreement, masked-text agreement and per-token log-prob differences.
- **Pass criterion.** At least 98.0% doc agreement (span and masked) and mean \|Δlogprob\| at most 0.23.
- **Where the limits come from.** The thresholds are measured noise floors, not guesses (`experiments/E1/README.md`):

  | comparison | doc agreement |
  |---|---|
  | same CPU oracle source, scalar vs AVX2 build (accumulation order only) | 98.96% |
  | bf16-emu CPU vs fp32 CPU | 97.59% (mean \|dlp\| 0.225) |
  | GPU at a different `--batch` | 99.61% |
  | deliberately wrong build (one layer skipped) | 45.31% (mean \|dlp\| 3.58) |

  The model is very sensitive to rounding, so exact equality with the CPU is not a meaningful target.
- **Gate scores of the shipped builds.**

  | build | doc agreement | mean \|dlp\| | result |
  |---|---|---|---|
  | `engine/` | 98.24% | 0.210 | pass |
  | `server/` default | 98.24% | 0.210 | pass |
  | `server/` BI | 98.57% | 0.208 | pass |
  | portable on A100 | 98.31% | 0.208 | pass |
  | FP8 opt-in | 95.50% | 0.525 | **fail** |

  `engine/` output is byte-identical (check.txt jsonl, masked text and log-probs, plus the mixed.txt md5) to `host_moe_attn_f16`, the build that first passed with fp16 attention. All later speedups changed no output bit.
- **Determinism.** No float atomics. Repeated runs give identical output, with one md5 across all runs of each build.
- **Host glue.** `bench/gate/replay_check.sh` feeds the CPU oracle's own log-probs through the CUDA-linked binary's real host pipeline. The output must be byte-identical to the oracle for several `--batch` sizes and thread counts, and this needs no GPU.
- **Upstream tokenizer bug.** Upstream pf calls `memmem` without `_GNU_SOURCE`. On glibc the pointer is truncated, so special tokens such as `<|endoftext|>` are **never matched**. All builds here compile with `-D_GNU_SOURCE`, and the oracle was rebuilt with that fix. Only check.txt doc 1532 is affected (`experiments/E1`, `experiments/E2`).

## Reproducing every number
The scripts are described in `bench/README.md`.

On a 1x H100 box: `bash /workspace/setup.sh && source /workspace/env.sh`. Each script is non-interactive and finishes in under 10 min. The data paths (`/workspace/bench/corpus/{mixed,check}.txt` and `/workspace/experiments/E1/oracle_bf16emu`) are absolute; override them with the env vars named in each script's header.

| result | command | evidence |
|---|---|---|
| **engine 2,650,119 tok/s** (gate, identity vs REF, paired ratios vs REF, f16_dbuf, stack_fd and the initial port) | `bash final/bench/batch/gpu_bench5.sh OUT` (`NAMES=stack_fdm REF=stack_fdm BASE= ` times the engine alone) | `experiments/E23/gpu_run5/summary.json`, `experiments/E25/confirm/` |
| server tables, BI invariance, gemm_bi microbench | `bash final/bench/serve/serve_bench2.sh OUT` | `experiments/E26/gpu_run/results.json` |
| A100 gate, E2E, MoE config sweep | `bash /workspace/candidates/gpu_bench_a100.sh OUT` (on an A100) | `experiments/E30/gpu_run/summary.json` |
| FP8 study and bf16-attention fast mode | `bash /workspace/candidates/gpu_bench_fp8.sh OUT` | `experiments/E23/gpu_run_fp8/summary.json` |
| attention precision study (7 variants) + attention microbench | `bash /workspace/candidates/gpu_bench3.sh OUT` | `experiments/E16/gpu_run/` |
| f16_fused, f16_dbuf, `--batch` sweep | `bash /workspace/candidates/gpu_bench4.sh OUT` | `experiments/E20/` |
| host pipeline + first fused MoE (1,110,486 and 1,533,448 tok/s) | `bash /workspace/candidates/gpu_bench.sh OUT` | `experiments/E9/` |
| initial port 458,886 tok/s, phase split, noise floors | `/workspace/implementation/pf_cuda/build.sh && cd /workspace/experiments/E1 && ./bench_mixed.sh 6` | `experiments/E1/` |
| MoE kernel microbench (E3 kernel vs cuBLAS) / MoE v2 | see `experiments/E3/README.md` "Reproduce"; `bash /workspace/experiments/E15/run_on_gpu.sh` | `experiments/E3/results.json`, `experiments/E18/gpu_run/` |
| upstream-linger vs adaptive server (E14) | `bash /workspace/experiments/E11/serve_bench.sh OUT` | `experiments/E14/results.json` |
| reference GPU 9,555 tok/s, CPU pf ~5,470 tok/s | commands in `experiments/E2/README.md` "Reproduce" | `experiments/E2/baselines.json` |
| gate on any binary | `bash final/bench/gate/gate.sh final/engine/pf_cuda OUT` (same runs and limits as `gpu_bench5.sh` step (ii); exit 1 on FAIL) | `OUT/gate.txt`, `OUT/vs_oracle.json` |
| replay identity (CPU-only) | `bash final/bench/gate/replay_check.sh OUT engine=final/engine/pf_cuda server=final/server/pf_cuda` | `experiments/E29/replay_check/results.txt` |
| multi-GPU scaling (not run) | `bash final/bench/scale/scale.sh OUT "1 2 4"` | none |

## Other platforms
See `PLATFORMS.md` for the tables.

- **A100-80GB (sm_80): measured.**
  - **Build:** `candidates/portable` is one fat binary with sm_80, sm_90a and sm_100 images. The sm_80 path uses E3's portable mma.sync grouped MoE in place of wgmma/TMA. The fp16 mma.sync attention, the elementwise kernels and the cuBLAS dense GEMMs are unchanged.
  - **Results:** **1,122,849 tok/s** E2E with `PF_MOE_CFG=4`, and 936,871 tok/s with the default config. It passes the gate at 98.31% / 0.208 (`experiments/E30`).
  - **H100 path:** its sm_90a SASS is identical to `engine/` (`experiments/E28`).
- **B200 (sm_100): not measured.** `candidates/b200` (and `candidates/portable`) contain sm_100 images that use the portable MoE. They were built and CPU-checked: the sm_90a SASS is identical to `engine/`'s and the replay check passes (`experiments/E24`). They were **never run**, because B200 was not available in this project.
- **Multi-GPU: not measured.** `bench/scale/scale.sh` runs one data-parallel process per GPU. It has been dry-run only, because each sandbox had 1 GPU.

## Limits and next levers
- **Host gap.** The forward alone runs at 2.91M tok/s, against 2.65M tok/s end to end, a 9% gap.
  - The remaining serial host work on the main thread is mainly the scatter of log-probs into documents, about 0.21 s of 2.78 s.
  - Moving the scatter into the workers or onto the GPU (on-device Viterbi) is the next host lever.
- **GPU time is dominated by the MoE.** The fused MoE takes 1.35 s of the 2.54 s forward, at 408–447 TF/s on H100 in microbenchmarks, against a ~1.2 s GEMM-only floor at 600 TF/s for the whole corpus (`REPORT.md`).
  - The next steps are an FP8 path that passes the gate (for example per-row h scales or keeping GEMM2 in bf16) and fusing GEMM1+GEMM2 per tile.
  - moe-reduce+norm takes 0.48 s.
- **Blackwell.** A tcgen05/UMMA MoE (TMEM accumulators, 2-CTA MMA) would replace the portable mma.sync path that `candidates/b200` uses on sm_100.
- **Batch-invariant cost.** BI costs 9% of E2E, because `gemm_bi` (mma.sync, fixed 128×128 tile, no split-K) is 1.4–3.5x slower than cuBLAS per GEMM. A wgmma fixed-tile variant with the same no-split structure should recover most of that while keeping invariance.
- **Server engine.** `server/` runs the f16_fused engine (2.42M tok/s batch in E20), not `engine/`'s MoE v2 kernel. Double buffering does not help the server, because it calls the synchronous `forward()`. The MoE v2 kernel can be swapped in following `experiments/E22/README.md` "Engine swap":
  1. take `engine/`'s `moe.cu`, `moe_ws.inc`, `ws_sched.h`, `moe_cfg.h` and `moe_sm90.inc`;
  2. keep the server's `pf.c` and `gemm_bi.*`;
  3. apply `tools/patch_attn_bi.py` and the `gpu_bi_vs_f16_fused.patch` hunks.

  This has not been built or measured.
- **Accuracy.** fp16 attention and bf16 activations put the engine at 98.24% doc agreement with the bf16-emulating CPU oracle. That is within the measured floors, but it is not bit-exact with CPU pf. Two more accurate builds are slower (E16 medians, `REPORT.md`):
  - `host_moe`, with fp32 SIMT attention, reaches 98.83% and is 1.53x slower;
  - split-bf16 attention reaches 98.89% and is 1.10x slower.
- **Upstream bug.** The memmem/_GNU_SOURCE bug above is fixed here but not upstream.
