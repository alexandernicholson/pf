# extras/fp8: opt-in FP8 (e4m3) expert GEMMs, which fail the correctness gate

This folder is `candidates/f16_fp8` with byte-identical sources (only this README was added). Its sources hash (sha12 over build.sh and the sources) to `d12bbc379099`, the id measured in `experiments/E23/gpu_run_fp8`. The design is in `experiments/E19/README.md`.

- **Base engine.** The engine is `host_moe_attn_f16`, i.e. fp16 tensor-core attention plus the E3 fused wgmma MoE. It does **not** include `../../engine`'s fused epilogue, double buffer or MoE v2, so it is slower than `engine/` even with FP8 on.
- **`PF_FP8_EXPERTS` unset (default).** The build is byte-identical to host_moe_attn_f16: the same check.txt outputs and the same mixed.txt md5. It ran at 2,343,037 tok/s and passes the gate at 98.24% / 0.210.
- **`PF_FP8_EXPERTS=1`.** The expert GEMMs run as e4m3 wgmma:
  - weight scales are per expert and per output column;
  - xh scales are per token and dynamic;
  - h uses a static 1/8 scale.

  `PF_FP8_CFG=g1,g2` picks the tiles; the default is `160s3p0,160s3p0`. Measured on 1x H100: 2,538,366 tok/s E2E, **1.090x** the exact path (geo mean of 5 paired rounds, CI [1.071, 1.110]). **The gate fails: 95.50% doc agreement and 0.525 mean |dlp|** (limits ≥98.0% and ≤0.23).

Build and run:
```sh
sh build.sh
PF_FP8_EXPERTS=1 ./pf_cuda --stats --lines -f docs.txt > masked.txt
```
