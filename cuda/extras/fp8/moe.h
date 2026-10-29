/* Fused grouped MoE expert block for pf's privacy-filter model (sm_80+).
 *
 * For each (token t, slot s) with e = eidx[t*4+s] >= 0:
 *   h1   = b1[e] + xh[t] @ w1[e]                       (fp32 accumulate, 1280 wide)
 *   h    = bf16_rne(swiglu(h1[:640], h1[640:]))        (gpt-oss clamps, alpha 1.702, +1 linear)
 *   out[t][s] = gates[t*4+s] * (b2[e] + h @ w2[e])     (fp32)
 * Slots with eidx < 0 (dynamic expert skipping) produce zero rows.
 * Output is fp32 [T][4][640]; the caller sums the 4 slots in order.
 * Deterministic: no float atomics; every output row depends only on its own inputs.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t moe_workspace_size(int T);

void moe_forward(const __nv_bfloat16 *xh, const int *eidx, const float *gates, int T,
                 const __nv_bfloat16 *w1, const __nv_bfloat16 *b1,
                 const __nv_bfloat16 *w2, const __nv_bfloat16 *b2,
                 float *out, void *workspace, cudaStream_t stream);

/* sm_90a path: weights pre-transposed once to K-major (w1t [128][1280][640], w2t [128][640][640]) */
void moe_prepare_weights(const __nv_bfloat16 *w1, const __nv_bfloat16 *w2, __nv_bfloat16 *w1t,
                         __nv_bfloat16 *w2t, cudaStream_t stream);
void moe_forward_sm90(const __nv_bfloat16 *xh, const int *eidx, const float *gates, int T,
                      const __nv_bfloat16 *w1t, const __nv_bfloat16 *b1,
                      const __nv_bfloat16 *w2t, const __nv_bfloat16 *b2,
                      float *out, void *workspace, cudaStream_t stream);

/* opt-in e4m3 path (see moe_fp8_sm90.inc): quantize the K-major bf16 weights once (w1q [128][1280][640] +
 * s1 [128][1280], w2q [128][640][640] + s2 [128][640]), then forward with per-token dynamic xh scales and a
 * static h scale. Changes outputs (approximation); the bf16 entry points above are unaffected. */
size_t moe_workspace_size_fp8(int T);
void moe_prepare_weights_fp8(const __nv_bfloat16 *w1t, const __nv_bfloat16 *w2t, uint8_t *w1q, float *s1,
                             uint8_t *w2q, float *s2, cudaStream_t stream);
int moe_fp8_set_cfg(const char *gemm1_cfg, const char *gemm2_cfg);   /* 0 ok, -1 unknown name */
void moe_forward_sm90_fp8(const __nv_bfloat16 *xh, const int *eidx, const float *gates, int T,
                          const uint8_t *w1q, const float *s1, const __nv_bfloat16 *b1,
                          const uint8_t *w2q, const float *s2, const __nv_bfloat16 *b2,
                          float *out, void *workspace, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
