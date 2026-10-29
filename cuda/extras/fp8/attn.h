/* attn.h — tensor-core banded attention with sinks for pf (sm_80+, tuned for sm_90a).
 *
 * Semantics (pf.c ph_b, same as the port's k_attn): for query t and head h
 * (kv group g = h / 7), keys j in [max(t-128, sst[t]), min(t+128, sen[t]-1)],
 * s_j = q_t . k_j (q, k already RoPE'd and scaled by 64^-0.25),
 * o = sum_j softmax({s_j} U {sinks[h]*ln2})_j v_j, the sink only in the
 * denominator. Output bf16 (RNE) [T][896].
 *
 * Input layout is the port's qkv buffer [T][1152]: q cols 0..895, k 896..1023,
 * v 1024..1151.
 *   attn_fwd_f32  — Variant A: fp32 qkv (the port's buffer), rounded on load to
 *                   the operand precision selected at compile time by ATTN_PREC.
 *   attn_fwd_bf16 — Variant B: bf16 qkv, same layout (for a fused rope epilogue);
 *                   always ATTN_PREC_BF16 math.
 * Deterministic: no atomics, fixed reduction order.
 *
 * ATTN_PREC (compile-time, default ATTN_PREC_BF16) picks the operand precision
 * of the two products; accumulation, softmax and l are fp32 in every variant:
 *   ATTN_PREC_BF16        QK^T bf16,       PV bf16        (E8, 8-bit significand)
 *   ATTN_PREC_F16         QK^T fp16,       PV fp16        (11-bit, same MMA rate)
 *   ATTN_PREC_QKBF_PVF16  QK^T bf16,       PV fp16        (ablation)
 *   ATTN_PREC_QKF16_PVBF  QK^T fp16,       PV bf16        (ablation)
 *   ATTN_PREC_SPLIT       QK^T split-bf16, PV split-bf16  (x = hi + lo, ~16-bit, 3 MMAs per product)
 *   ATTN_PREC_SPLITQK_PVF16  QK^T split-bf16, PV fp16     (QK rounding dominates: scores reach |s| ~ 340)
 * fp16 operands are converted with satfinite (|x| > 65504 clamps instead of
 * becoming Inf).
 */
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#define ATTN_PREC_BF16       0
#define ATTN_PREC_F16        1
#define ATTN_PREC_QKBF_PVF16 2
#define ATTN_PREC_QKF16_PVBF 3
#define ATTN_PREC_SPLIT      4
#define ATTN_PREC_SPLITQK_PVF16 5
#ifndef ATTN_PREC
#define ATTN_PREC ATTN_PREC_BF16
#endif

/* dynamic shared memory of the bf16/fp16 variants (split needs 180752 B,
 * splitqk_pvf16 135696 B, both run 1 CTA per SM; the launchers set the
 * attribute themselves) */
#define ATTN_SMEM_BYTES 90640

void attn_fwd_f32(const float *qkv, const int *sst, const int *sen, const float *sinks,
                  uint16_t *ao, int T, cudaStream_t st);
void attn_fwd_bf16(const uint16_t *qkv, const int *sst, const int *sen, const float *sinks,
                   uint16_t *ao, int T, cudaStream_t st);
