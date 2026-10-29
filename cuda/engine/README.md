> **final/engine** is `candidates/stack_fdm` with byte-identical sources (sha12 `17370154d25f`, the id measured in `experiments/E23/gpu_run5` and `experiments/E25/confirm`). Build: `sh build.sh` (needs sm_90a, CUDA 12.x). **Verified on 1x H100: 2,650,119 tok/s E2E on mixed.txt, gate PASS 98.24% / 0.210, output byte-identical to host_moe_attn_f16.** Usage, flags and results: `../README.md`, `../REPORT.md`. The candidate notes below are kept as written before the GPU run.

# stack_fdm — stack_fd + MoE v2 (E21)

**Status (at the time of writing): candidate, not yet run on a GPU.** Predicted byte-identical to `candidates/host_moe_attn_f16` (REF); confirmed on H100 in E23/E25.

This is `candidates/stack_fd` with the MoE GEMMs integrated the way `candidates/host_moe_v2` did it.

**Files from `experiments/E15/kernel/`:** `moe.cu`, `moe_sm90.inc`, `wgmma.inc`, `moe_ws.inc` and `ws_sched.h`, unchanged.

**`moe_cfg.h`** selects `moe::CfgWS<160, 6, 160, 6, true>`. That is E18's microbench pick (`experiments/E18/best_cfg.txt`): the warp-specialized persistent kernel with TMA weight loads, BN = 160 and 6 stages for both GEMMs. On H100 it was bitwise equal to E3's kernel and deterministic on real routing.

**TMA descriptor.** It is encoded on the host in `moe_ws.inc` through `cudaGetDriverEntryPoint("cuTensorMapEncodeTiled")`, so no `-lcuda` is needed. The descriptor is cached per weight pointer.

**Other files are unchanged from stack_fd:**
- `moe.h`, so the API and `moe_workspace_size` are the same;
- `gpu.cu`, except one header comment;
- `pf.c`, `attn.cu` and `attn.h`.

**`build.sh`** is stack_fd's plus `-Xptxas -v` on `moe.cu`, which writes `moe_ptxas.log`. That log shows 0 spill bytes and 0 stack bytes in all 8 entries, including `k_ws<1,160,6,true>` and `k_ws<2,160,6,true>` (168 registers).
