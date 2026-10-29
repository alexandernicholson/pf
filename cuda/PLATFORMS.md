# Platforms

All GPU numbers come from scripted single-GPU runs on mixed.txt (47,928 docs, 7,371,164 tokens), E2E as defined in README.md. Gate: check.txt vs `experiments/E1/oracle_bf16emu`, ≥98.0% doc agreement and ≤0.23 mean |dlp|.

| GPU | build | MoE path | E2E tok/s (median, n=5) | gate | evidence |
|---|---|---|---|---|---|
| H100 80GB HBM3 (sm_90a) | `final/engine` (= `candidates/stack_fdm`) | wgmma + TMA, warp-specialized persistent | **2,650,119** (fresh box: 2,648,663) | PASS 98.24% / 0.210 | `experiments/E23/gpu_run5`, `experiments/E25/confirm` |
| A100-SXM4-80GB (sm_80) | `candidates/portable` (fat binary), `PF_MOE_CFG=0` | portable mma.sync, 128×128 tiles, 3 stages | 936,871 (932,853–937,088) | PASS 98.31% / 0.208 | `experiments/E30/gpu_run/summary.json` |
| A100-SXM4-80GB (sm_80) | `candidates/portable`, **`PF_MOE_CFG=4`** | portable mma.sync, 128×128 tiles, 2 stages | **1,122,849** (3 runs; 1.20x paired vs cfg 0; output byte-identical to cfg 0) | same output as cfg 0 | `experiments/E30/gpu_run/summary.json` (`cfg_sweep`) |
| B200 (sm_100) | `candidates/b200`, `candidates/portable` | portable mma.sync | not measured: B200 is not available to this project | — | `experiments/E24` (CPU checks only) |

## A100 details (`experiments/E30`)

**Running it.** Run the portable fat binary on an A100 with `PF_MOE_CFG=4`. On compute capability 8.0 it selects the portable MoE automatically. On H100 the same binary runs the unchanged sm_90a path.

**MoE microbench** (synthetic skewed routing, layer 0):

| T | cfg 0 (128×128, s3) | cfg 2 (g1 128×256) | cfg 4 (128×128, s2) |
|---|---|---|---|
| 65536 | 115 TF/s | 132 TF/s | 155 TF/s |
| 131072 | 117 TF/s | 135 TF/s | **159 TF/s** |

- All configs give identical output hashes.
- A100 dense bf16 peak is 312 TF/s.

**GPU phase split (cfg 0, ms per corpus):**

| phase | ms |
|---|---|
| MoE | 4,929 |
| moe-reduce+norm | 835 |
| attention | 686 |
| qkv gemm | 577 |
| out gemm | 300 |
| bias+rope | 255 |

The MoE dominates even more than on H100.

**A100 vs H100.** With cfg 4 the A100 reaches about 42% of the H100 engine.

**Next levers on A100:**
- make cfg 4 the default on CC 8.0;
- an A100-tuned MoE kernel with larger K tiles and an ldmatrix-friendly B layout.

## Multi-GPU
`bench/scale/scale.sh` (= `candidates/scale.sh`) runs one data-parallel process per GPU on contiguous chunks of documents, merges outputs in order and compares them with the single-GPU output. It has been dry-run only, because this project's sandboxes are limited to 1 GPU.

Expect near-linear scaling: documents are independent and host work is ~35M tok/s on 8 cores. Merged output will be ~99% doc-identical to N=1, because chunking changes batch composition; the batch-invariant server build removes that difference.
