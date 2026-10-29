/* gpu.cu — CUDA forward pass for pf: one packed batch of tokens -> log-probs.
 *
 * Mirrors pf.c's forward() with the BFDOT numerics: fp32 residual stream and
 * norms, activations rounded to bf16 (RNE) at every matmul input, bf16
 * weights, fp32 accumulation, biases added in fp32. Dense projections are
 * cuBLAS bf16 GEMMs with fp32 output; the MoE is E3's fused grouped wgmma
 * kernel (moe.cu: on-device routing, W1 + bias + SwiGLU + bf16, W2 + bias,
 * x gate, out[t][slot][D] fp32), so there is no host sync per layer.
 * Attention is E8's tensor-core kernel (attn.cu, Variant A: reads the fp32
 * qkv buffer, rounds q/k/v to bf16 on load, bf16 P, fp32 accumulation) over
 * the +-BAND window clipped to each packed segment; PF_ATTN_SIMT=1 selects
 * the previous tiled fp32 kernel (k_attn) for diagnosis. The expert
 * reduction is done per token in slot order (no atomics on the residual),
 * so a given input gives bit-identical log-probs run to run.
 * PF_FP8_EXPERTS=1 (opt-in approximation, changes outputs) runs the experts as e4m3 wgmma GEMMs instead
 * (moe_fp8_sm90.inc: per-channel weight scales, per-token xh scales, static h scale); PF_FP8_CFG=g1,g2
 * picks its tile configs. Unset, the forward is the bf16 path, unchanged.
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "moe.h"
#include "attn.h"

#define D      640
#define NL     8
#define NH     14
#define NKV    2
#define HD     64
#define QD     (NH * HD)
#define KVD    (NKV * HD)
#define QKVD   (QD + 2 * KVD)
#define NE     128
#define TOPK   4
#define FF     640
#define NC     33
#define BAND   128
#define SWIGLU_ALPHA 1.702f
#define SWIGLU_LIMIT 7.0f
#define RMS_EPS      1e-5f
#define QK_SCALE     0.35355339059327379f

typedef uint16_t bf16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "pf: CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "pf: cuBLAS status %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ float b2f(bf16 h) { return __uint_as_float((uint32_t)h << 16); }
/* round-to-nearest-even, same bit trick as pf's cvt_f32_bf16 */
__device__ __forceinline__ bf16 f2b(float f) {
    uint32_t u = __float_as_uint(f);
    return (bf16)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* ------------------------------------------------------------------ state */
typedef struct { const void *p; size_t n; } GpuT;   /* host tensor + bytes (pf.c order) */

typedef struct {
    const bf16 *attn_norm, *qkv_w, *qkv_b; const float *sinks;
    const bf16 *out_w, *out_b, *mlp_norm, *gate_w, *gate_b, *w1, *b1, *w2, *b2;
} GLayer;

static struct { const bf16 *emb, *fnorm, *unemb; GLayer L[NL]; } W;

static struct {
    int cap, n_ctx, skip_layer, attn_simt;
    int *ids, *pos, *sst, *sen;     /* [cap] */
    float *rc, *rs;                 /* [n_ctx][HD/2] */
    float *x;                       /* [cap][D] residual */
    bf16 *tbh;                      /* [cap][D] normed, bf16 */
    float *qkv;                     /* [cap][QKVD] q|k|v, rope'd and scaled in place */
    bf16 *ao;                       /* [cap][QD] attention out */
    float *tmp;                     /* [cap][D] out-proj / router logits */
    int *eidx; float *gates;        /* [cap][TOPK] */
    float *mo;                      /* [cap][TOPK][D] gate * (b2 + h @ w2), 0 if skipped */
    void *mws;                      /* moe workspace (routing, tiles, h) */
    bf16 *w1t[NL], *w2t[NL];        /* K-major expert weights for moe_forward_sm90 */
    int fp8;                        /* PF_FP8_EXPERTS=1: e4m3 experts (opt-in approximation) */
    uint8_t *w1q[NL], *w2q[NL];     /* e4m3 K-major expert weights + per-channel scales */
    float *s1[NL], *s2[NL];
    float *lp_d, *lp_h;             /* [cap][NC]; lp_h pinned */
    cublasHandle_t cb;
} G;

/* per-phase GPU time from events recorded between kernels */
enum { P_COPY, P_EMBED, P_NORM, P_QKV, P_ROPE, P_ATTN, P_OUT, P_ROUTER, P_TOPK,
       P_MOE, P_HEAD, NPH };
static const char *PH_NAME[NPH] = {"h2d/d2h", "embed", "moe-reduce+norm", "qkv gemm",
    "bias+rope", "attention", "out gemm", "router gemm", "topk",
    "moe fused (route+gemm1+swiglu+gemm2)", "head"};
#define NEV 256
static cudaEvent_t ev[NEV];
static int ev_ph[NEV], nev;
static double ph_gpu[NPH];
static void mark(int ph) { ev_ph[nev] = ph; CK(cudaEventRecord(ev[nev++], 0)); }

/* --------------------------------------------------------------- kernels */
__global__ void k_embed(const int *ids, const bf16 *emb, float *x, int T) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)T * D) return;
    size_t t = i / D, c = i % D;
    x[i] = b2f(emb[(size_t)ids[t] * D + c]);
}

/* warp per token: x += a + ab (out-proj) or x += sum_s mo[t][s] in slot order
 * (moe; mo = gate*(b2 + h@w2)), then tbh = bf16(rmsnorm(x) * scale) */
__global__ void k_resid_norm(float *x, const float *a, const bf16 *ab,
                             const float *mo, const int *eidx,
                             const bf16 *scale, bf16 *tbh, int T) {
    int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    float v[D / 32];
    float *xr = x + (size_t)t * D;
#pragma unroll
    for (int j = 0; j < D / 32; j++) v[j] = xr[lane + 32 * j];
    if (a) {
        const float *ar = a + (size_t)t * D;
#pragma unroll
        for (int j = 0; j < D / 32; j++) v[j] += ar[lane + 32 * j] + b2f(ab[lane + 32 * j]);
    }
    if (mo) {
        for (int s = 0; s < TOPK; s++) {
            int it = t * TOPK + s;
            if (eidx[it] < 0) continue;               /* skipped by --skip-beta */
            const float *o = mo + (size_t)it * D;
#pragma unroll
            for (int j = 0; j < D / 32; j++) v[j] += o[lane + 32 * j];
        }
    }
    float ss = 0;
#pragma unroll
    for (int j = 0; j < D / 32; j++) ss += v[j] * v[j];
    ss = warp_sum(ss);
    float r = 1.0f / sqrtf(ss / D + RMS_EPS);
    bf16 *th = tbh + (size_t)t * D;
#pragma unroll
    for (int j = 0; j < D / 32; j++) {
        int c = lane + 32 * j;
        xr[c] = v[j];
        th[c] = f2b(v[j] * r * b2f(scale[c]));
    }
}

/* qkv bias; interleaved-pair rope on q and k heads; QK_SCALE on q and k */
__global__ void k_rope(float *qkv, const bf16 *bias, const float *rc, const float *rs,
                       const int *pos, int T) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const int per = (QD + KVD) / 2 + KVD;             /* 512 pairs + 128 v */
    if (i >= (size_t)T * per) return;
    size_t t = i / per; int r = (int)(i % per);
    float *row = qkv + t * QKVD;
    if (r < (QD + KVD) / 2) {
        int c = 2 * r, f = r % (HD / 2);
        float cs = rc[pos[t] * (HD / 2) + f], sn = rs[pos[t] * (HD / 2) + f];
        float x1 = row[c] + b2f(bias[c]), x2 = row[c + 1] + b2f(bias[c + 1]);
        row[c]     = (x1 * cs - x2 * sn) * QK_SCALE;
        row[c + 1] = (x2 * cs + x1 * sn) * QK_SCALE;
    } else {
        int c = QD + KVD + (r - (QD + KVD) / 2);
        row[c] += b2f(bias[c]);
    }
}

/* banded attention with sinks, fp32. Block = 64 queries x one head; keys are
 * swept in 64-key chunks over the union of the tile's windows (each window is
 * +-BAND clipped to the query's segment), with an online softmax seeded by the
 * sink (m = sink, l = exp(0) = 1). Thread (ty,tx) owns query rows ty*4..+3
 * and key/dim columns tx*4..+3. */
#define AQ 64
#define AK 64
#define ATT_SMEM ((HD * AQ + HD * AK + AK * HD + AK * (AQ + 4)) * 4)
__global__ __launch_bounds__(256) void k_attn(const float *qkv, const int *sst, const int *sen,
                                             const float *sinks, bf16 *ao, int T) {
    extern __shared__ float sm[];
    float *Qs = sm;                 /* [HD][AQ]   */
    float *Ks = Qs + HD * AQ;       /* [HD][AK]   */
    float *Vs = Ks + HD * AK;       /* [AK][HD]   */
    float *Ps = Vs + AK * HD;       /* [AK][AQ+4] */
    __shared__ int lo_s[AQ], hi_s[AQ], kr[2];
    const int h = blockIdx.y, g = h / (NH / NKV), t0 = blockIdx.x * AQ, tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    if (tid < AQ) {
        int t = t0 + tid, lo = 0, hi = -1;
        if (t < T) { lo = max(t - BAND, sst[t]); hi = min(t + BAND, sen[t] - 1); }
        lo_s[tid] = lo; hi_s[tid] = hi;
    }
    for (int i = tid; i < AQ * HD / 4; i += 256) {
        int r = i & (AQ - 1), d = (i / AQ) * 4, t = t0 + r;
        float4 v = make_float4(0, 0, 0, 0);
        if (t < T) v = *(const float4 *)(qkv + (size_t)t * QKVD + h * HD + d);
        Qs[d * AQ + r] = v.x; Qs[(d + 1) * AQ + r] = v.y;
        Qs[(d + 2) * AQ + r] = v.z; Qs[(d + 3) * AQ + r] = v.w;
    }
    __syncthreads();
    if (tid == 0) {
        int a = INT_MAX, b = -1;
        for (int r = 0; r < AQ; r++)
            if (hi_s[r] >= lo_s[r]) { a = min(a, lo_s[r]); b = max(b, hi_s[r]); }
        kr[0] = a; kr[1] = b;
    }
    __syncthreads();
    const int kmin = kr[0], kmax = kr[1];
    int lo[4], hi[4];
    float m[4], l[4], o[4][4];
    const float sink = sinks[h] * (float)M_LN2;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        lo[i] = lo_s[ty * 4 + i]; hi[i] = hi_s[ty * 4 + i];
        m[i] = sink; l[i] = 1.0f;
#pragma unroll
        for (int j = 0; j < 4; j++) o[i][j] = 0;
    }
    for (int k0 = kmin; k0 <= kmax; k0 += AK) {
        for (int i = tid; i < AK * HD / 4; i += 256) {
            int r = i & (AK - 1), d = (i / AK) * 4, j = k0 + r;
            float4 v = make_float4(0, 0, 0, 0);
            if (j < T) v = *(const float4 *)(qkv + (size_t)j * QKVD + QD + g * HD + d);
            Ks[d * AK + r] = v.x; Ks[(d + 1) * AK + r] = v.y;
            Ks[(d + 2) * AK + r] = v.z; Ks[(d + 3) * AK + r] = v.w;
        }
        for (int i = tid; i < AK * HD / 4; i += 256) {
            int r = i / (HD / 4), d = (i % (HD / 4)) * 4, j = k0 + r;
            float4 v = make_float4(0, 0, 0, 0);
            if (j < T) v = *(const float4 *)(qkv + (size_t)j * QKVD + QD + KVD + g * HD + d);
            *(float4 *)(Vs + r * HD + d) = v;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) s[i][j] = 0;
#pragma unroll 16
        for (int d = 0; d < HD; d++) {
            float4 qa = *(const float4 *)(Qs + d * AQ + ty * 4);
            float4 kb = *(const float4 *)(Ks + d * AK + tx * 4);
            float qv[4] = {qa.x, qa.y, qa.z, qa.w}, kv[4] = {kb.x, kb.y, kb.z, kb.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) s[i][j] = fmaf(qv[i], kv[j], s[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float mx = m[i];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int jj = k0 + tx * 4 + j;
                if (jj < lo[i] || jj > hi[i]) s[i][j] = -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            float alpha = expf(m[i] - mx), ps = 0;
            m[i] = mx;
#pragma unroll
            for (int j = 0; j < 4; j++) { s[i][j] = expf(s[i][j] - mx); ps += s[i][j]; }
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
            l[i] = l[i] * alpha + ps;
#pragma unroll
            for (int j = 0; j < 4; j++) o[i][j] *= alpha;
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) Ps[(tx * 4 + j) * (AQ + 4) + ty * 4 + i] = s[i][j];
        __syncthreads();
#pragma unroll 16
        for (int k = 0; k < AK; k++) {
            float4 pa = *(const float4 *)(Ps + k * (AQ + 4) + ty * 4);
            float4 vb = *(const float4 *)(Vs + k * HD + tx * 4);
            float pv[4] = {pa.x, pa.y, pa.z, pa.w}, vv[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) o[i][j] = fmaf(pv[i], vv[j], o[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int t = t0 + ty * 4 + i;
        if (t >= T) continue;
        float inv = 1.0f / l[i];
        bf16 *dst = ao + (size_t)t * QD + h * HD + tx * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) dst[j] = f2b(o[i][j] * inv);
    }
}

/* warp per token: router bias, top-4 (strict >, lowest index wins ties, as
 * pf), softmax over the 4; optional --skip-beta */
__global__ void k_topk(const float *lg, const bf16 *gb, int *eidx, float *gates,
                       int T, float beta) {
    int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    float g[NE / 32];
#pragma unroll
    for (int k = 0; k < NE / 32; k++) g[k] = lg[(size_t)t * NE + lane + 32 * k] + b2f(gb[lane + 32 * k]);
    int ei[TOPK]; float gw[TOPK];
#pragma unroll
    for (int s = 0; s < TOPK; s++) {
        float bv = -1e30f; int bi = NE;
#pragma unroll
        for (int k = 0; k < NE / 32; k++) if (g[k] > bv) { bv = g[k]; bi = lane + 32 * k; }
        for (int off = 16; off; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi == NE) bi = 0;
        ei[s] = bi; gw[s] = bv;
#pragma unroll
        for (int k = 0; k < NE / 32; k++) if (lane + 32 * k == bi) g[k] = -1e30f;
    }
    float mx = gw[0], den = 0;
    for (int s = 0; s < TOPK; s++) { gw[s] = expf(gw[s] - mx); den += gw[s]; }
    for (int s = 0; s < TOPK; s++) gw[s] /= den;
    if (beta > 0.0f) {
        float thr = beta * gw[0], kden = 0;
        int keep = TOPK;
        for (int s = 0; s < TOPK; s++) {
            if (s > 0 && gw[s] < thr) { keep = s; break; }
            kden += gw[s];
        }
        for (int s = keep; s < TOPK; s++) ei[s] = -1;
        for (int s = 0; s < keep; s++) gw[s] /= kden;
    }
    if (lane == 0)
        for (int s = 0; s < TOPK; s++) {
            eidx[t * TOPK + s] = ei[s]; gates[t * TOPK + s] = gw[s];
        }
}

/* warp per token: moe reduce, final norm, 33-class head, log_softmax */
#define HEAD_TOK 64
__global__ void k_head(float *x, const float *mo, const int *eidx, const bf16 *fnorm,
                       const bf16 *unemb, float *lp, int T) {
    __shared__ bf16 U[NC * D];
    for (int i = threadIdx.x; i < NC * D; i += blockDim.x) U[i] = unemb[i];
    __syncthreads();
    int lane = threadIdx.x & 31, nw = blockDim.x / 32;
    for (int k = threadIdx.x / 32; k < HEAD_TOK; k += nw) {
        int t = blockIdx.x * HEAD_TOK + k;
        if (t >= T) break;
        float v[D / 32];
        float *xr = x + (size_t)t * D;
#pragma unroll
        for (int j = 0; j < D / 32; j++) v[j] = xr[lane + 32 * j];
        for (int s = 0; s < TOPK; s++) {
            int it = t * TOPK + s;
            if (eidx[it] < 0) continue;
            const float *o = mo + (size_t)it * D;
#pragma unroll
            for (int j = 0; j < D / 32; j++) v[j] += o[lane + 32 * j];
        }
        float ss = 0;
#pragma unroll
        for (int j = 0; j < D / 32; j++) ss += v[j] * v[j];
        ss = warp_sum(ss);
        float r = 1.0f / sqrtf(ss / D + RMS_EPS);
#pragma unroll
        for (int j = 0; j < D / 32; j++) v[j] = b2f(f2b(v[j] * r * b2f(fnorm[lane + 32 * j])));
        float lg[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float a = 0;
#pragma unroll
            for (int j = 0; j < D / 32; j++) a = fmaf(v[j], b2f(U[c * D + lane + 32 * j]), a);
            lg[c] = warp_sum(a);
        }
        float mx = -1e30f;
#pragma unroll
        for (int c = 0; c < NC; c++) mx = fmaxf(mx, lg[c]);
        float den = 0;
#pragma unroll
        for (int c = 0; c < NC; c++) den += expf(lg[c] - mx);
        float lden = logf(den) + mx;
        float *out = lp + (size_t)t * NC;
#pragma unroll
        for (int c = 0; c < NC; c++) if ((c & 31) == lane) out[c] = lg[c] - lden;
    }
}

/* ------------------------------------------------------------------ gemms */
/* C[M][N] fp32 = X[M][K] . Wt[N][K]^T  (pf's dense weights are [out][in]) */
static void gemm_nt(const bf16 *X, const bf16 *Wt, float *C, int M, int N, int K) {
    const float one = 1.0f, zero = 0.0f;
    CB(cublasGemmEx(G.cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, Wt, CUDA_R_16BF, K,
                    X, CUDA_R_16BF, K, &zero, C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

/* -------------------------------------------------------------------- api */
#define DMALLOC(p, n) CK(cudaMalloc((void **)&(p), (size_t)(n) * sizeof *(p)))
static int nblk(size_t n, int b) { return (int)((n + b - 1) / b); }

extern "C" float *gpu_init(int cap, int n_ctx, const GpuT *t, int nt,
                           const float *rc, const float *rs) {
    if (nt != 3 + 13 * NL) { fprintf(stderr, "pf: gpu_init: bad tensor list\n"); exit(1); }
    size_t tot = 0;
    for (int i = 0; i < nt; i++) tot += (t[i].n + 255) & ~(size_t)255;
    uint8_t *arena;
    CK(cudaMalloc((void **)&arena, tot));
    const void *dp[3 + 13 * NL];
    for (int i = 0; i < nt; i++) {        /* one aligned copy per tensor */
        CK(cudaMemcpy(arena, t[i].p, t[i].n, cudaMemcpyHostToDevice));
        dp[i] = arena; arena += (t[i].n + 255) & ~(size_t)255;
    }
    W.emb = (const bf16 *)dp[0]; W.fnorm = (const bf16 *)dp[1]; W.unemb = (const bf16 *)dp[2];
    for (int l = 0; l < NL; l++) {
        const void *const *q = dp + 3 + 13 * l;
        GLayer *y = &W.L[l];
        y->attn_norm = (const bf16 *)q[0]; y->qkv_w = (const bf16 *)q[1];
        y->qkv_b = (const bf16 *)q[2];     y->sinks = (const float *)q[3];
        y->out_w = (const bf16 *)q[4];     y->out_b = (const bf16 *)q[5];
        y->mlp_norm = (const bf16 *)q[6];  y->gate_w = (const bf16 *)q[7];
        y->gate_b = (const bf16 *)q[8];    y->w1 = (const bf16 *)q[9];
        y->b1 = (const bf16 *)q[10];       y->w2 = (const bf16 *)q[11];
        y->b2 = (const bf16 *)q[12];
    }
    G.cap = cap; G.n_ctx = n_ctx;
    size_t ni = (size_t)cap * TOPK;
    DMALLOC(G.ids, cap); DMALLOC(G.pos, cap); DMALLOC(G.sst, cap); DMALLOC(G.sen, cap);
    DMALLOC(G.rc, (size_t)n_ctx * HD / 2); DMALLOC(G.rs, (size_t)n_ctx * HD / 2);
    CK(cudaMemcpy(G.rc, rc, (size_t)n_ctx * HD / 2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G.rs, rs, (size_t)n_ctx * HD / 2 * 4, cudaMemcpyHostToDevice));
    DMALLOC(G.x, (size_t)cap * D); DMALLOC(G.tbh, (size_t)cap * D);
    DMALLOC(G.qkv, (size_t)cap * QKVD); DMALLOC(G.ao, (size_t)cap * QD);
    DMALLOC(G.tmp, (size_t)cap * D);
    DMALLOC(G.eidx, ni); DMALLOC(G.gates, ni);
    DMALLOC(G.mo, ni * D);
    G.fp8 = getenv("PF_FP8_EXPERTS") && atoi(getenv("PF_FP8_EXPERTS"));
    CK(cudaMalloc(&G.mws, G.fp8 ? moe_workspace_size_fp8(cap) : moe_workspace_size(cap)));
    for (int l = 0; l < NL; l++) {        /* K-major copies; checkpoint layout stays (unused) */
        DMALLOC(G.w1t[l], (size_t)NE * D * 2 * FF); DMALLOC(G.w2t[l], (size_t)NE * FF * D);
        moe_prepare_weights((const __nv_bfloat16 *)W.L[l].w1, (const __nv_bfloat16 *)W.L[l].w2,
                            (__nv_bfloat16 *)G.w1t[l], (__nv_bfloat16 *)G.w2t[l], 0);
    }
    if (G.fp8) {
        for (int l = 0; l < NL; l++) {
            DMALLOC(G.w1q[l], (size_t)NE * 2 * FF * D); DMALLOC(G.s1[l], (size_t)NE * 2 * FF);
            DMALLOC(G.w2q[l], (size_t)NE * D * FF);     DMALLOC(G.s2[l], (size_t)NE * D);
            moe_prepare_weights_fp8((const __nv_bfloat16 *)G.w1t[l], (const __nv_bfloat16 *)G.w2t[l],
                                    G.w1q[l], G.s1[l], G.w2q[l], G.s2[l], 0);
        }
        const char *c = getenv("PF_FP8_CFG");
        char g1[32] = "160s3p0", g2[32] = "160s3p0";
        if (c && *c && sscanf(c, "%31[^,],%31s", g1, g2) != 2) { fprintf(stderr, "pf: PF_FP8_CFG=g1,g2\n"); exit(1); }
        if (moe_fp8_set_cfg(g1, g2)) { fprintf(stderr, "pf: unknown PF_FP8_CFG %s,%s\n", g1, g2); exit(1); }
        fprintf(stderr, "pf: PF_FP8_EXPERTS=1: e4m3 expert GEMMs (approximate), gemm1 %s gemm2 %s\n", g1, g2);
    }
    CK(cudaGetLastError());
    DMALLOC(G.lp_d, (size_t)cap * NC);
    CK(cudaMallocHost((void **)&G.lp_h, (size_t)cap * NC * 4));
    CB(cublasCreate(&G.cb));
    CK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, ATT_SMEM));
    attn_fwd_f32(NULL, NULL, NULL, NULL, NULL, 0, 0);   /* T = 0: only sets the smem attribute */
    G.attn_simt = getenv("PF_ATTN_SIMT") && atoi(getenv("PF_ATTN_SIMT"));
    for (int i = 0; i < NEV; i++) CK(cudaEventCreate(&ev[i]));
    G.skip_layer = getenv("PF_GPU_SKIP_LAYER") ? atoi(getenv("PF_GPU_SKIP_LAYER")) : -1;
    CK(cudaDeviceSynchronize());
    return G.lp_h;
}

/* one packed batch -> G.lp_h (the buffer gpu_init returned), synchronous */
extern "C" void gpu_forward(const int32_t *ids, const int *pos, const int *sst, const int *sen,
                            int T, float skip_beta) {
    nev = 0;
    mark(P_COPY);
    CK(cudaMemcpyAsync(G.ids, ids, (size_t)T * 4, cudaMemcpyHostToDevice, 0));
    CK(cudaMemcpyAsync(G.pos, pos, (size_t)T * 4, cudaMemcpyHostToDevice, 0));
    CK(cudaMemcpyAsync(G.sst, sst, (size_t)T * 4, cudaMemcpyHostToDevice, 0));
    CK(cudaMemcpyAsync(G.sen, sen, (size_t)T * 4, cudaMemcpyHostToDevice, 0));
    mark(P_EMBED);
    k_embed<<<nblk((size_t)T * D, 256), 256>>>(G.ids, W.emb, G.x, T);
    int moe_pending = 0;                  /* mo holds an unreduced moe output */
    for (int l = 0; l < NL; l++) {
        if (l == G.skip_layer) continue;  /* debug: deliberately wrong model */
        const GLayer *y = &W.L[l];
        mark(P_NORM);
        k_resid_norm<<<nblk(T, 8), 256>>>(G.x, NULL, NULL, moe_pending ? G.mo : NULL, G.eidx,
                                          y->attn_norm, G.tbh, T);
        mark(P_QKV);
        gemm_nt(G.tbh, y->qkv_w, G.qkv, T, QKVD, D);
        mark(P_ROPE);
        k_rope<<<nblk((size_t)T * ((QD + KVD) / 2 + KVD), 256), 256>>>(G.qkv, y->qkv_b, G.rc, G.rs, G.pos, T);
        mark(P_ATTN);
        if (G.attn_simt)
            k_attn<<<dim3(nblk(T, AQ), NH), 256, ATT_SMEM>>>(G.qkv, G.sst, G.sen, y->sinks, G.ao, T);
        else
            attn_fwd_f32(G.qkv, G.sst, G.sen, y->sinks, G.ao, T, 0);
        mark(P_OUT);
        gemm_nt(G.ao, y->out_w, G.tmp, T, D, QD);
        mark(P_NORM);
        k_resid_norm<<<nblk(T, 8), 256>>>(G.x, G.tmp, y->out_b, NULL, NULL,
                                          y->mlp_norm, G.tbh, T);
        mark(P_ROUTER);
        gemm_nt(G.tbh, y->gate_w, G.tmp, T, NE, D);
        mark(P_TOPK);
        k_topk<<<nblk(T, 8), 256>>>(G.tmp, y->gate_b, G.eidx, G.gates, T, skip_beta);
        mark(P_MOE);
        if (G.fp8)
            moe_forward_sm90_fp8((const __nv_bfloat16 *)G.tbh, G.eidx, G.gates, T, G.w1q[l], G.s1[l],
                                 (const __nv_bfloat16 *)y->b1, G.w2q[l], G.s2[l], (const __nv_bfloat16 *)y->b2,
                                 G.mo, G.mws, 0);
        else
            moe_forward_sm90((const __nv_bfloat16 *)G.tbh, G.eidx, G.gates, T,
                             (const __nv_bfloat16 *)G.w1t[l], (const __nv_bfloat16 *)y->b1,
                             (const __nv_bfloat16 *)G.w2t[l], (const __nv_bfloat16 *)y->b2, G.mo, G.mws, 0);
        moe_pending = l + 1;
    }
    mark(P_HEAD);
    if (!moe_pending) { fprintf(stderr, "pf: no layers run\n"); exit(1); }
    k_head<<<nblk(T, HEAD_TOK), 256>>>(G.x, G.mo, G.eidx, W.fnorm, W.unemb, G.lp_d, T);
    mark(P_COPY);
    CK(cudaMemcpyAsync(G.lp_h, G.lp_d, (size_t)T * NC * 4, cudaMemcpyDeviceToHost, 0));
    mark(P_COPY);
    CK(cudaStreamSynchronize(0));
    CK(cudaGetLastError());
    for (int i = 0; i + 1 < nev; i++) {
        float ms; CK(cudaEventElapsedTime(&ms, ev[i], ev[i + 1]));
        ph_gpu[ev_ph[i]] += ms;
    }
}

extern "C" void gpu_stats(FILE *f) {
    double tot = 0;
    for (int p = 0; p < NPH; p++) tot += ph_gpu[p];
    fprintf(f, "gpu phases (ms, event-timed incl. host gaps):");
    for (int p = 0; p < NPH; p++) fprintf(f, " %s %.1f |", PH_NAME[p], ph_gpu[p]);
    fprintf(f, " total %.1f\n", tot);
}
