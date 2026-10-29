# pf-cuda technical report

**Scope.** This report covers the CUDA port of pf ([upstream](https://github.com/alexandernicholson/pf), a one-file C implementation of OpenAI's `privacy-filter` PII model) and its hill-climb toward the fastest NVIDIA serving platform.

**Hardware.** Every throughput number was measured on 1x NVIDIA H100 80GB HBM3 (driver 610.57.04, CUDA 12.8) unless marked otherwise.

**Sources.** Every number in this report appears in the cited evidence file, and paths are relative to `/workspace`. Ratios computed here, rather than read from a file, are marked *(computed)*.

## 1. Method

**Workloads.**
- **mixed.txt** is the headline workload: 47,928 docs and 7,371,164 o200k tokens, one document per line, run with `pf_cuda --stats --lines -f mixed.txt`.
- **check.txt** is the correctness workload: 1,534 docs and 353,505 tokens, including 24 long multi-window documents and edge cases.

**Headline metric: E2E tok/s.** It is tokens divided by the `--stats` "e2e" wall time, which covers tokenize, forward, Viterbi decode, span selection and output. It excludes process start, file read and model load plus weight upload (about 0.75 s; `experiments/E23/gpu_run5/timing/*.err`).

**Forward tok/s** uses GPU-synchronized forward time, H2D/D2H included. It is reported for attribution only.

**Timing protocol.** It is the same for every scripted GPU run (`gpu_bench*.sh`):
- Each binary is built from a copied source tree on the GPU box.
- The check.txt gate runs **before** any timing, and a binary that fails the gate is not timed, or is timed and marked FAIL.
- Timing is 1 warmup plus 5 rounds over mixed.txt, with the order of binaries rotated each round. Medians are reported.
- Speedups are **paired ratios within a round** (candidate / reference in the same round), summarized by median or geometric mean with min/max or a 95% CI. With n = 5 the exact one-sided sign test cannot go below p = 1/32, so significance rests on the paired t-interval of the log ratios (`experiments/E20/README.md`, `experiments/E23/paired_eval.txt`).
- Output md5s are recorded per run.

**Correctness gate.** The candidate is compared with a CPU oracle on check.txt using `bench/compare.py` (copied unchanged to `final/bench/gate/compare.py`).
- **Oracle:** upstream pf with bf16 rounding of matmul inputs, the semantics pf validated against the reference, and with the tokenizer fix below (`experiments/E1/oracle_bf16emu`).
- **Pass:** at least 98.0% per-document span agreement and masked-text agreement, and mean |Δlogprob| at most 0.23.
- **Why these limits** (`experiments/E1/README.md`). The model is very sensitive to rounding, so the limits are set from measured floors:

  | comparison | doc agreement | mean \|dlp\| |
  |---|---|---|
  | the CPU oracle's own source, scalar vs AVX2/FMA build (accumulation order only) | 98.96% | |
  | bf16-emu CPU vs fp32 CPU | 97.59% | 0.225 |
  | the GPU at two `--batch` sizes | 99.61% | 0.066 |
  | the reference PyTorch GPU vs CPU pf (`experiments/E2`) | 96.35% | |
  | deliberately wrong build (one block skipped) | 45.31% | 3.58 |

  The deliberately wrong build is rejected by a wide margin.

**Host-glue identity.** `PF_NULL_FORWARD=replay:` feeds the oracle's own log-probs through a CUDA-linked binary's real host pipeline. The output must be byte-identical to the oracle at several batch sizes and thread counts. This runs on a CPU box and isolates host-side bugs from numerics (`experiments/E10`, `E13`, `E17`, `E21`, `E24`, `E29`).

**Upstream bug found.** pf.c calls `memmem` without `_GNU_SOURCE`. With `gcc -std=gnu11` on glibc, `memmem` is implicitly declared `int` and the returned pointer is truncated, so special tokens (`<|endoftext|>`, `<|endofprompt|>`) are **never matched**. Every build here adds `-D_GNU_SOURCE`, and the oracle was rebuilt with the fix. Only check.txt doc 1532 is affected (13 tokens upstream vs 7 fixed; `experiments/E1/README.md`, `experiments/E2/README.md`).

## 2. Feasibility bound

The bound follows from the model config (`/workspace/.cache/pf_model/config.json`): 8 layers, d = 640, 14 query heads and 2 KV heads of 64, 128 experts with top-4, expert width 640 (GLU, so W1 is 640→1280), sliding window 257 (±128) and 33 labels. Multiply-adds per token *(computed)*:

| part | MFLOP/token (2·MAC) |
|---|---|
| experts: 8 layers × 4 × 2·(640·1280 + 640·640) | 78.6 |
| qkv 640→1152, out 896→640, router 640→128: 8 × 2·(737,280 + 573,440 + 81,920) | 22.3 |
| attention: QKᵀ + PV over ≤257 keys, 14 heads × 64: 8 × 2·2·14·64·257 | 7.4 |
| head 640→33 | 0.04 |
| **total** | **≈108** |

**GEMM-only floor.** The dense and expert GEMMs are 100.9 MFLOP/token, which over 7,371,164 tokens is 7.44·10¹⁴ FLOP. At 600 TF/s (a good Hopper bf16 GEMM, about 60% of the 989 TF dense peak) that takes **≈1.24 s per corpus**, i.e. about 5.9M tok/s if nothing else cost time.

**Where we are.** The best build's forward takes 2.54 s (2.91M tok/s) and its E2E is 2.78 s (2.65M tok/s). That is about 2x the GEMM floor. The gap is the MoE kernel at 408–447 TF/s instead of 600, plus about 1.05 s of non-GEMM phases (section 4): attention 0.29 s, moe-reduce + norm 0.48 s, bias + rope 0.15 s, and smaller ones *(computed from the E23 phase medians)*.

## 3. Optimizations, in the order they were stacked

All rows are mixed.txt E2E on 1x H100. The paired ratio compares each build with the build named in the "vs" column, within the same run.

| # | build (`candidates/…`) | what changed | E2E tok/s (median) | paired ratio (vs) | gate (agree / \|dlp\|) | evidence |
|---|---|---|---|---|---|---|
| ref | reference GPU, openai/privacy-filter PyTorch + Triton, `torch.compile` | one `predict_text` per doc, n_ctx 4096 | 9,555 | | 96.74% vs CPU pf (compile build, check.txt) | `experiments/E2/baselines.json` |
| ref | CPU pf, 8 threads | upstream | ~5,470 (mixed first 5k docs) | | oracle | `experiments/E2/README.md` |
| 1 | initial port (`implementation/pf_cuda`) | GPU forward: cuBLAS GEMMs, fp32 SIMT attention, 128 cuBLAS calls per expert GEMM; serial host | 458,886 (E1, n=6); 487,859 (E9); 486,542 (E23, n=3) | | 98.96% / 0.154 | `experiments/E1/results.json`, `experiments/E9/gpu_run/summary.json` |
| 2 | `host` | parallel host pipeline: threaded tokenize and Viterbi, overlapped with the GPU | 1,110,486 | 2.280 geo, CI [2.264, 2.295], 5/5 (vs 1) | 98.96% (output identical to 1) | `experiments/E9/results.json` |
| 3 | `host_moe` | fused grouped MoE: on-device routing, wgmma GEMM1 + SwiGLU epilogue + GEMM2 (E3 kernel) | 1,533,448 (E9); 1,529,713 (E16) | 1.379 geo, rounds 1.374–1.385, 5/5 (vs 2) *(computed from E9 trials)* | 98.83% / 0.152 | `experiments/E9/`, `experiments/E16/gpu_run/` |
| 4 | `host_moe_attn_f16` | tensor-core attention (mma.sync m16n8k16, ldmatrix, cp.async), **fp16** operands | 2,344,739 (E16); 2,351,401 (E20) | 1.530 median, [1.518, 1.534], 5/5 (vs 3) | **98.24% / 0.210** | `experiments/E16/gpu_run/paired_stats.txt` |
| 5 | `f16_fused` | qkv GEMM epilogue fused with bias + RoPE + fp16 conversion; attention variant B reads fp16 q\|k\|v | 2,420,193 | 1.033 geo, CI [1.021, 1.045] (vs 4) | identical to 4 | `experiments/E20/paired_stats.json` |
| 6 | `f16_dbuf` | double-buffered forward (`gpu_submit`/`gpu_wait`, D2H on a copy stream): batch k+1 packs and scatters while k runs | 2,496,491 (E20); 2,516,432 (E23) | 1.061 geo, CI [1.052, 1.069] (vs 4, E20); 1.076 geo (vs 4, E23) | identical to 4 | `experiments/E20/`, `experiments/E23/paired_eval.txt` |
| 7 | `stack_fd` | 5 + 6 | 2,600,128 | 1.034 median, geo 1.035, CI [1.029, 1.041], 5/5 (vs 6) | identical to 4 | `experiments/E23/gpu_run5/summary.json` |
| 8 | **`stack_fdm` = `final/engine`** | MoE v2: warp-specialized persistent wgmma kernel with TMA weight loads, `CfgWS<160,6,160,6,true>` (E15/E18) | **2,650,119** (E23); **2,648,663** (E25, fresh box) | 1.019 median, [1.013, 1.024], 5/5 (vs 7); 1.054 geo, CI [1.053, 1.055] (vs 6) | identical to 4 | `experiments/E23/gpu_run5/summary.json`, `experiments/E25/results.json` |

**Totals.**
- `stack_fdm` vs the initial port: **5.446x** median paired (3 pairs, range 5.425–5.482) in E23, and 5.471x in E25.
- `stack_fdm` vs `host_moe_attn_f16`: 1.132x (E23) and 1.139x (E25), 5/5 pairs each time.
- `stack_fdm` vs the reference GPU implementation: 277x *(computed, 2,650,119 / 9,555; the reference was measured with its natural per-document usage, `experiments/E2/README.md`)*.

**Identity.** For builds 5–8, check.txt jsonl, masked text and log-prob dumps are byte-identical to build 4, and every mixed.txt run has md5 `452071c9…`. None of these speedups changed an output bit.

**Batch size.** `--batch` 262144 and 524288 gave 1.003x and 1.007x and changed numerics, so the default stays at 131072 (`experiments/E20/README.md`).

**Fresh-box confirmation** (E25, `experiments/E25/results.json`):

| build | E2E median |
|---|---|
| initial | 484,111 |
| host_moe_attn_f16 | 2,327,384 |
| f16_dbuf | 2,487,891 |
| stack_fd | 2,574,050 |
| stack_fdm | 2,648,663 |

The order and ratios match E23.

## 4. Where the time goes

GPU phase medians (ms per mixed.txt pass, cudaEvent-timed):

| phase | initial port (E1) | host_moe (E16) | host_moe_attn_f16 (E16) | **stack_fdm (E23)** |
|---|---|---|---|---|
| attention | 1,991 | 1,992.6 | 295.6 | **286.6** |
| expert GEMMs + SwiGLU + routing | 1,230 + 844 + 819 + 294 | 1,329.3 (fused) | 1,369.6 | **1,349.6** (MoE v2) |
| moe-reduce + norm | 510 | 479.7 | 479.9 | 479.7 |
| qkv gemm | 213 | 230.9 | 235.1 | 225.8 |
| bias + rope | 226 | 225.2 | 227.0 | 146.9 (fused epilogue) |
| out gemm | 121 | 119.6 | 129.2 | 126.7 |
| router gemm / topk | 43 / – | 44.3 / 32.5 | 44.2 / 34.2 | 43.9 / 35.3 |
| head, embed, h2d/d2h | 52, –, 49 | 46.3, 16.7, 46.7 | 47.3, 17.1, 46.2 | 48.1, 17.9, 30.7 |

**stack_fdm main thread** (`experiments/E23/gpu_run5/summary.json`): forward 2,535.9 ms, scatter 209.5 ms, pack 16.0 ms, write 19.4 ms, out of an E2E of 2,781 ms. Forward-only throughput is 2,906,685 tok/s against 2,650,119 E2E. The remaining 9% is mostly the serial log-prob scatter.

## 5. Kernel-level results

### 5.1 Fused grouped MoE (E3, E15, E18)

**Design.**
- Routing is fully on device: a histogram, a scan, and a scatter into expert segments padded to 128 rows, with no host sync.
- GEMM1 gathers token rows with cp.async and fuses bias + SwiGLU + bf16 rounding in registers. The B tile pairs glu and lin columns.
- GEMM2 fuses bias, gate and a scatter to `[T][4][640]` fp32.
- There are no float atomics, so results are bit-identical from run to run (`experiments/E3/README.md`).

**E3** (layer-0 weights, Zipf-skewed routing; `experiments/E3/README.md`):

| T | cuBLAS per-expert loop | mma.sync 128×128 | wgmma 128×160 s3 (shipped in host_moe) | vs cuBLAS |
|---|---|---|---|---|
| 16,384 | 74 TF/s | 239 | 341 | 4.64x |
| 65,536 | 152 | 273 | 421 | 2.78x |
| 131,072 | 186 | 276 | 412 | 2.22x |
| 262,144 | 210 | 277 | 408 | 1.94x |

**E18** (MoE v2 on H100, `experiments/E18/gpu_run/select.txt`, `results.json`):
- Every warp-specialized persistent variant was **bitwise equal to the E3 kernel** and deterministic.
- At T = 131,072 the variants reached 425.5–446.9 TF/s.
- On real routing (T = 131,002, layers 0 and 4) the E3 kernel reached 409.4 and 418.0 TF/s.

| variant (T = 131,072) | TF/s | GEMM1 TF/s | GEMM2 TF/s |
|---|---|---|---|
| E3 wgmma 160s3 | 434.9 | 457.4 | 383.1 |
| ws cp.async 128s6 | 425.8 | 442.6 | 413.9 |
| ws TMA 128s6 | 425.5 | 461.2 | 393.0 |
| ws TMA 160s5 | 427.2 | 472.7 | 438.6 |
| **ws TMA 160s6 (selected)** | **446.9** | 454.1 | 441.4 |

In the E2E run, MoE v2 gave 1.019x E2E over stack_fd. It is 43–45% of the 989 TF dense peak: K = 640 gives only 10 k-steps per tile, so prologue and epilogue latency matter.

### 5.2 Tensor-core attention (E8, E12, E13, E16)

**Design.**
- The attention is banded and bidirectional (±128) with per-head sinks, over packed multi-document batches with per-segment bounds.
- It uses mma.sync m16n8k16, with K/V tiles in shared memory via cp.async and ldmatrix(.trans), and an online softmax seeded with the sink.
- It uses 90,640 B of shared memory, so 2 CTAs fit per SM. It has 121 registers and 0 spills (`experiments/E13/README.md`).

**Results.**
- **Phase time per corpus:** 1,992.6 ms with fp32 SIMT, 295.6 ms with fp16 TC and 286.6 ms with fp16 variant B plus the fused epilogue (section 4), a 7.0x reduction.
- **Microbench, T ≥ 65,536, useful TF/s** (masked work excluded):

  | kernel | run | useful TF/s | projected per corpus |
  |---|---|---|---|
  | bf16 (E8 kernel) | E12 | 160–201 | 0.24–0.29 s |
  | fp16 (E13 kernel) | E16 | 130–163 | 0.31–0.36 s |
  | SIMT port | both | 18–21 | |

  Sources: `experiments/E12/gpu_run/attn_bench/results.json`, `experiments/E16/gpu_run/attn_bench/results.json`.

### 5.3 Attention precision study (E13 design, E16 measurement)

bf16 attention was fast but failed the gate. A range probe on the real model found Σ|q·k| up to 461 in layer 0, so the QKᵀ rounding error in the scores drives the failure (`experiments/E13/README.md`). Seven precision variants were gated and the passing ones timed (`experiments/E16/gpu_run/check/gate.txt`, `paired_stats.txt`):

| variant (QKᵀ / PV operands) | doc agree | mean \|dlp\| | gate | E2E tok/s | x host_moe (paired) |
|---|---|---|---|---|---|
| fp32 SIMT (host_moe) | 98.83% | 0.152 | PASS | 1,529,713 | 1 |
| bf16 / bf16 (`host_moe_attn`) | 97.52% | 0.287 | FAIL | 2,358,885 | 1.546 |
| bf16 / fp16 | 97.72% | 0.277 | FAIL | not timed | |
| fp16 / bf16 | 97.85% | 0.239 | FAIL | not timed | |
| **fp16 / fp16 (`host_moe_attn_f16`, shipped)** | **98.24%** | **0.210** | **PASS** | **2,344,739** | **1.530** |
| split-bf16 QK / fp16 PV | 98.44% | 0.202 | PASS | 2,176,332 | 1.423 |
| split-bf16 / split-bf16 (3 MMAs) | 98.89% | 0.169 | PASS | 2,127,681 | 1.383 |

**Findings.**
- QK precision decides the gate: fp16 QK cuts |dlp| from 0.277 to 0.239, while fp16 PV alone barely moves it.
- fp16 for both products passes at bf16 speed, with the same MMA rate and shared memory.
- The split variants are more accurate but run at 1 CTA/SM.
- **bf16 attention as an opt-in fast mode was dropped.** In E23 it was only 1.015x faster than fp16 attention (median paired; geo 1.018x; 2,379,527 vs 2,336,481 tok/s, `experiments/E23/gpu_run_fp8`) and it fails the gate.

## 6. FP8 experts study (E19 design, E23 measurement)

**Design** (`candidates/f16_fp8`, packaged as `final/extras/fp8`).
- With `PF_FP8_EXPERTS=1`, the expert GEMMs run as e4m3 wgmma.
- Weights use per-expert, per-output-column scales, and activations use per-token dynamic scales.
- h uses a static 1/8 scale, bounded analytically by the SwiGLU clamps.
- With the variable unset, the build is byte-identical to `host_moe_attn_f16`: 100% agreement with it and the same mixed md5.

**Results** (`experiments/E23/gpu_run_fp8/summary.json`, `experiments/E23/paired_eval.txt`):

| mode | E2E tok/s | forward tok/s | MoE phase | doc agree | mean \|dlp\| | argmax flips | gate |
|---|---|---|---|---|---|---|---|
| f16_fp8, default (exact) | 2,343,037 | 2,537,054 | 1,366.2 ms | 98.24% | 0.210 | 112 | PASS |
| **f16_fp8, `PF_FP8_EXPERTS=1`** | 2,538,366 | 2,782,642 | 1,132.6 ms | **95.50%** | **0.525** | 338 | **FAIL** |

**Findings.**
- FP8 vs exact on the same binary: **1.090x** geo, CI [1.071, 1.110], 5/5.
- Compared with the exact path, FP8 agrees on only 95.63% of docs.
- FP8 makes the MoE phase 17% faster, but the E2E gain is smaller than the stacked v2 + dbuf + fused gains. So FP8 stays **opt-in and off by default**, and it was not stacked onto `stack_fdm`.
- Making FP8 pass would need finer h scaling, or bf16 for GEMM2 only. Neither was tried.

## 7. Server (E11 design, E14/E26 measurement)

**Design** (`candidates/serve2`, packaged as `final/server`). This is pf's `--serve` on the GPU engine (f16_fused):
- One batcher thread owns the CUDA context. Connection threads tokenize and decode.
- An **adaptive linger** dispatches immediately when the engine was busy or when no other connection is mid-read.
- Each job has its own wake-up.
- HTTP/1.1 keep-alive, one `writev` per response, TCP_NODELAY and a 4096-entry backlog.
- `GET /stats` reports the batcher counters.

**Load test protocol.**
- Closed-loop clients over 4 load-generator processes, 12 s runs plus a 2 s warmup, docs from mixed.txt.
- **Every response** is checked against that build's own `--lines` batch output.
- An exact check sends 500 docs in one request before and after the load and requires byte identity; it was 500/500 identical both times in both builds.

E26 (`experiments/E26/gpu_run/results.json`; responses verified: 488,091 for BI, all identical; 488,317 for default, 484,044 identical):

| run | default docs/s | default tok/s | default p50 / p99 ms | BI docs/s | BI tok/s | BI p50 / p99 ms |
|---|---|---|---|---|---|---|
| C=1, docs ≤64 tok | 461.3 | 26,604 | 2.16 / 2.50 | 417.3 | 24,065 | 2.40 / 2.70 |
| C=1 | 413.1 | 63,438 | 2.30 / 6.04 | 374.3 | 57,809 | 2.60 / 2.93 |
| C=8 | 1,278.6 | 206,883 | 5.09 / 15.35 | 1,393.2 | 219,458 | 5.61 / 9.62 |
| C=64 | 5,703.6 | 895,254 | 10.85 / 20.83 | 7,353.4 | 1,121,635 | 7.94 / 17.52 |
| C=256 | 11,614.0 | 1,789,374 | 20.89 / 39.72 | 11,646.7 | 1,793,339 | 20.15 / 41.35 |
| 16 clients × 64-doc requests | 15,978.7 | 2,454,594 | 61.30 / 90.46 | 14,032.0 | 2,145,991 | 69.60 / 101.92 |
| responses identical to batch path | **99.12%** | | | **100%** | | |
| batch-mode E2E (mixed.txt) | 2,389,631 | | | 2,169,359 (0.908x paired, rounds 0.901–0.933) | | |
| check.txt gate | 98.24% / 0.210 PASS | | | 98.57% / 0.208 PASS | | |

**Batch invariance.**
- **What BI changes.** The BI build replaces cuBLAS for the qkv, out and router GEMMs with `gemm_bi` (mma.sync, one fixed 128×128×32 tile, no split-K, fixed K order), and aligns attention key steps to segment starts (`-DATTN_BI`). Everything else was already per token or per row.
- **Invariance test.** 50 target docs were packed 4 ways with fillers, offsets and `--batch` 131072/8192/4096. The BI build gave **0 differing fp32 log-prob bits**. The default build differed in 13,603–13,908 tokens (max |Δlp| 15.7), which explains its 99.12% server agreement.
- **Why BI costs 9%.** `gemm_bi` is 1.4–3.5x slower than cuBLAS on these shapes, for example qkv at M = 131,072 takes 0.852 vs 0.355 ms. It is bitwise equal to cuBLAS on the accuracy rows (`gemm_bench` in the same file).
- **Why BI is faster at C=64.** Mid-concurrency latency and throughput are better with BI (C=64: 7,353 vs 5,704 docs/s). This was measured but not explained; the batch-composition sensitivity of cuBLAS kernel selection at small M is a plausible cause that was not tested.

**Linger policy** (E14, host_moe engine, `experiments/E14/results.json`):
- Upstream's fixed 8 ms linger gave **90.4 docs/s at C=1** (p50 10.7 ms), against 431.3 docs/s (p50 2.2 ms) with the adaptive policy.
- At C=64 it gave 3,472 vs 4,819 docs/s.

## 8. Limitations
- **Platforms measured: H100 (headline) and A100-80GB.** This project could only use H100 and A100-80GB, with 1 GPU per sandbox.
  - **A100-80GB:** the portable fat binary (`candidates/portable`, sm_80/sm_90a/sm_100) reaches **1,122,849 tok/s** E2E with `PF_MOE_CFG=4` (936,871 with the default config). It passes the gate at 98.31% / 0.208. Evidence and details: `experiments/E30/gpu_run/summary.json`, `final/PLATFORMS.md`.
  - **B200:** `candidates/b200` (fat binary with sm_90a and sm_100 images) is built and CPU-verified, with sm_90a SASS identical to the engine and replay identity. It has **never run**.
  - **Multi-GPU:** data-parallel scaling (`final/bench/scale/scale.sh`) has been dry-run only.
- **Per-document disagreement audit.** All 27 check docs where the engine differs from the CPU oracle are explained. 23 also differ between two valid CPU builds or under a GPU batch-size change. 3 are Viterbi-only, with no argmax flip. 1 is a single flip with a 0.089-nat margin. The largest flip margin is 2.12, against 3.23 between valid CPU builds. See `final/DISAGREEMENTS.md`.
- **Not bit-exact with CPU pf.** fp16 attention and bf16 activations give 98.24% doc agreement with the bf16-emulating oracle. That is inside the measured floors, but users who need maximum agreement should use a split-attention build (98.89%, 1.10x slower).
- **The default server is not batch-invariant.** 0.9% of responses differ from the batch path. `PF_BATCH_INVARIANT=1` fixes this at 9% batch throughput.
- **The server runs an older engine.** It uses f16_fused, not MoE v2 or the double buffer. MoE v2 would add about 2% (the stack_fdm vs stack_fd ratio); double buffering does not apply to the synchronous server path. The swap is documented but neither built nor measured.
- **FP8 fails the gate** and stays opt-in. bf16 attention was dropped.
- **Built with `-march=native`.** Binaries must be built on the host that runs them.

## 9. Next levers (ranked by expected value)
1. **Host scatter and decode.** Forward is 2.91M tok/s against 2.65M E2E. Parallelizing or moving to the GPU the 0.21 s serial scatter, and optionally running Viterbi on device, is worth up to about 9%.
2. **MoE toward 600 TF/s.** Fuse GEMM1 and GEMM2 per 128-row tile, keeping h in shared memory, use 2-CTA clusters with TMA multicast of weights, and fold the 4-slot reduce (0.48 s) into GEMM2's epilogue. The GEMM-bound floor is about 1.24 s, against the current forward of 2.54 s.
3. **Blackwell.** A tcgen05/UMMA MoE with TMEM accumulators would replace the portable mma.sync MoE that `candidates/b200` uses on sm_100.
4. **wgmma batch-invariant GEMM.** It would keep the fixed-tile, no-split structure and should recover most of BI's 9% cost.
5. **Server on MoE v2.** Follow `experiments/E22/README.md` "Engine swap".
6. **FP8 that passes the gate.** Use per-row h scales, or FP8 for GEMM1 only.
