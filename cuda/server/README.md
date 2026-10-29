> **final/server** is `candidates/serve2` with byte-identical sources (sha12 `d57e388c49dc` over pf.c, gpu.cu, attn.cu, gemm_bi.cu, moe.cu, the id measured in `experiments/E26/gpu_run`). **Verified on 1x H100** (both builds pass the gate; BI responses 100% identical to the batch path; C=256: 11,614 docs/s default, 11,647 BI). Usage: `../README.md` "HTTP server"; results: `../REPORT.md` section 7. Engine: f16_fused (not `../engine`'s MoE v2); swapping it in is described in `experiments/E22/README.md` "Engine swap" and has not been done. The candidate notes below are kept as written before the GPU run.

# serve2 — pf `--serve` (E11 server) on the f16_fused engine, with an optional batch-invariant build (E22)

**Status (at the time of writing): candidate, not yet run on a GPU.** Measured on H100 in E26. Details, CPU checks and the GPU run script are in `experiments/E22/README.md`.

- **Default build** (`sh build.sh`).
  - `pf.c` is `candidates/serve/pf.c` (E11's server, with dynamic batching and keep-alive) plus `--dump-lpbits`.
  - The engine is `candidates/f16_fused`. The `gpu.o` and `attn.o` SASS are identical to f16_fused's.
- **Batch-invariant build** (`PF_BATCH_INVARIANT=1 sh build.sh`). A document's log-probs are bit-identical whatever else is packed in the batch:
  - `gemm_bi.cu` (fixed-tile mma.sync, no split-K) replaces cuBLAS for the qkv, out and router GEMMs;
  - `attn.cu` is compiled with `-DATTN_BI`, which aligns key steps to the segment start;
  - everything else is already per token or per row.

  Its numerics differ slightly from the default build's, so it has its own check.txt gate.
- **GPU run:** `bash /workspace/experiments/E22/serve_bench2.sh`.
