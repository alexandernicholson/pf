/* attn.cu — tensor-core banded attention with sinks for pf; see attn.h.
 *
 * CTA = one 64-query tile x one kv group (all 7 q heads of the group), 7 warps:
 * warp w computes head g*7+w for the tile's 64 rows, as 4 row groups of 16.
 * The K/V rows of the tile's key union (<= 64 + 2*128 = 320 rows, plus a
 * zeroed 32-row slack) are converted once into swizzled shared memory and
 * reused by all 7 heads, so K/V global traffic is per group, not per head.
 *
 * Per row group the warp sweeps its own key union [jlo, jhi] (<= 272 keys) in
 * 32-key steps: S = Q K^T with mma.sync m16n8k16 (fp32 acc), mask by the
 * per-row window [lo, hi] (skipped for steps inside every row's window),
 * online softmax in fp32 seeded by the sink (m = sink, l = 1), P rounded
 * straight from the S accumulators into A fragments, O += P V.
 * l is accumulated from the fp32 probabilities; o / l is rounded once (RNE).
 *
 * Operand precision is a template pair (QP for Q/K, VP for P/V), each one of
 *   AT_BF    bf16 (E8)
 *   AT_F16   fp16, same fragment layouts and MMA rate as bf16
 *   AT_SPLIT x = hi + lo with hi = bf16(x), lo = bf16(x - hi); a product is
 *            a_lo*b_hi + a_hi*b_lo + a_hi*b_hi (3 bf16 MMAs, a_lo*b_lo dropped).
 *            hi and lo each get their own shared-memory plane, so a split
 *            stream doubles its K or V smem and the kernel runs 1 CTA per SM.
 * ATTN_PREC (attn.h) selects the pair behind attn_fwd_f32.
 *
 * Built with -DATTN_EMU the same source compiles for the host emulator
 * (emu_rt.h), which runs every CUDA thread as a host thread.
 */
#include "attn.h"

#ifdef ATTN_EMU
#include "emu_rt.h"
#else
#include <limits.h>

#define AT_DYN_SMEM(name) extern __shared__ __align__(128) unsigned char name[]

__device__ __forceinline__ void at_ldsm4(uint32_t *r, const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void at_ldsm4t(uint32_t *r, const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void at_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void at_mma_h(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/* two floats -> bf16x2, RNE; lo goes to the low half (lower column) */
__device__ __forceinline__ uint32_t at_pack(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
/* two floats -> f16x2, RNE, |x| > 65504 saturates to +-65504 (no Inf) */
__device__ __forceinline__ uint32_t at_pack_h(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}
__device__ __forceinline__ float at_u2f(uint32_t u) { return __uint_as_float(u); }
__device__ __forceinline__ float at_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float at_shfl_xor(float v, int o) { return __shfl_xor_sync(0xffffffffu, v, o); }
__device__ __forceinline__ int at_rmin(int v) { return __reduce_min_sync(0xffffffffu, v); }
__device__ __forceinline__ int at_rmax(int v) { return __reduce_max_sync(0xffffffffu, v); }
#define AT_SYNC() __syncthreads()
#endif

#define AT_NH     14
#define AT_HPG    7                     /* q heads per kv group */
#define AT_HD     64
#define AT_QD     (AT_NH * AT_HD)
#define AT_KVD    (2 * AT_HD)
#define AT_QKVD   (AT_QD + 2 * AT_KVD)
#define AT_BAND   128
#define AT_Q      64                    /* queries per CTA */
#define AT_KS     32                    /* keys per softmax step */
#define AT_ROWS   (AT_Q + 2 * AT_BAND + AT_KS)   /* staged K/V rows incl. zero slack */
#define AT_THREADS (AT_HPG * 32)
#define AT_LN2    0.69314718055994531f
#define AT_L2E    1.44269504088896341f

/* operand precisions */
#define AT_BF    0
#define AT_F16   1
#define AT_SPLIT 2

/* ATTN_PREC -> (QP, VP) */
__host__ __device__ constexpr int at_qp(int prec) {
    return prec == ATTN_PREC_F16 || prec == ATTN_PREC_QKF16_PVBF       ? AT_F16
           : prec == ATTN_PREC_SPLIT || prec == ATTN_PREC_SPLITQK_PVF16 ? AT_SPLIT
                                                                         : AT_BF;
}
__host__ __device__ constexpr int at_vp(int prec) {
    return prec == ATTN_PREC_F16 || prec == ATTN_PREC_QKBF_PVF16 || prec == ATTN_PREC_SPLITQK_PVF16 ? AT_F16
           : prec == ATTN_PREC_SPLIT                                                          ? AT_SPLIT
                                                                                              : AT_BF;
}

/* shared memory: K planes, then V planes ([AT_ROWS][64] 16-bit each, 128-byte
 * rows, 16-byte chunk c of row r stored at chunk c ^ (r & 7)); a split stream
 * has a hi plane followed by a lo plane. Then lo[64], hi[64], kr[4]. */
template <int QP, int VP> struct AtLay {
    static constexpr int PL = AT_ROWS * 128;
    static constexpr int K = 0, KL = PL;
    static constexpr int V = (QP == AT_SPLIT ? 2 : 1) * PL, VL = V + PL;
    static constexpr int LO = V + (VP == AT_SPLIT ? 2 : 1) * PL;
    static constexpr int HI = LO + AT_Q * 4, KR = HI + AT_Q * 4, END = KR + 16;
    /* CTAs per SM that fit H100's 228 KB (1 KB reserved per CTA) */
    static constexpr int MINB = 2 * (END + 1024) <= 228 * 1024 ? 2 : 1;
};
static_assert(AtLay<AT_BF, AT_BF>::END == ATTN_SMEM_BYTES, "ATTN_SMEM_BYTES out of date");
static_assert(AtLay<AT_F16, AT_F16>::MINB == 2 && AtLay<AT_SPLIT, AT_SPLIT>::END <= 227 * 1024, "smem budget");

__device__ __forceinline__ int at_swz(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

template <int P> __device__ __forceinline__ uint32_t at_cvt(float lo, float hi) {
    return P == AT_F16 ? at_pack_h(lo, hi) : at_pack(lo, hi);
}
/* split residual of a packed hi = at_pack(lo, hi): bf16(x - hi) per half
 * (x - bf16(x) is exact in fp32) */
__device__ __forceinline__ uint32_t at_res(float lo, float hi, uint32_t h) {
    return at_pack(lo - at_u2f(h << 16), hi - at_u2f(h & 0xFFFF0000u));
}
template <int P> __device__ __forceinline__ void at_mmap(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    if (P == AT_F16) at_mma_h(d, a, b0, b1);
    else at_mma(d, a, b0, b1);
}

/* 8 consecutive input elements -> one 16-byte smem chunk (h), plus the split
 * residual chunk (l) when P == AT_SPLIT */
template <int P> __device__ __forceinline__ void at_ld8(const float *p, uint4 &h, uint4 &l) {
    float4 x = __ldg((const float4 *)p), y = __ldg((const float4 *)(p + 4));
    h = make_uint4(at_cvt<P>(x.x, x.y), at_cvt<P>(x.z, x.w), at_cvt<P>(y.x, y.y), at_cvt<P>(y.z, y.w));
    if (P == AT_SPLIT)
        l = make_uint4(at_res(x.x, x.y, h.x), at_res(x.z, x.w, h.y), at_res(y.x, y.y, h.z), at_res(y.z, y.w, h.w));
}
template <int P> __device__ __forceinline__ void at_ld8(const uint16_t *p, uint4 &h, uint4 &) {
    static_assert(P == AT_BF, "bf16 input only feeds bf16 operands");
    h = __ldg((const uint4 *)p);
}
/* 2 consecutive input elements -> packed pair (h), plus split residual (l) */
template <int P> __device__ __forceinline__ void at_ld2(const float *p, uint32_t &h, uint32_t &l) {
    float2 x = __ldg((const float2 *)p);
    h = at_cvt<P>(x.x, x.y);
    if (P == AT_SPLIT) l = at_res(x.x, x.y, h);
}
template <int P> __device__ __forceinline__ void at_ld2(const uint16_t *p, uint32_t &h, uint32_t &) {
    static_assert(P == AT_BF, "bf16 input only feeds bf16 operands");
    h = __ldg((const unsigned int *)p);
}

/* one 32-key step for one warp's 16 query rows (rows ra = g, rb = g + 8 of the
 * fragment, g = lane / 4). MASK=false only when every key is inside every
 * row's window. q/ql: Q fragments (ql = split residual, QP == AT_SPLIT only);
 * Ks/Vs point at the hi planes, the lo planes follow at +AT_ROWS*128. */
template <bool MASK, int QP, int VP>
__device__ __forceinline__ void at_step(const unsigned char *Ks, const unsigned char *Vs, int rbase, int j0,
                                        const uint32_t (*q)[4], const uint32_t (*ql)[4], float (*o)[4],
                                        float &m_a, float &m_b, float &l_a, float &l_b, int lo_a, int hi_a,
                                        int lo_b, int hi_b, int lane) {
    constexpr int PL = AT_ROWS * 128;
    float s[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int half = 0; half < 2; half++) {
            uint32_t b[4];
            const int off = at_swz(rbase + nt * 8 + (lane & 7), half * 4 + (lane >> 3));
            at_ldsm4(b, Ks + off);
            if (QP == AT_SPLIT) {   /* small terms first */
                uint32_t bl[4];
                at_ldsm4(bl, Ks + PL + off);
                at_mma(s[nt], ql[half * 2 + 0], b[0], b[1]);
                at_mma(s[nt], q[half * 2 + 0], bl[0], bl[1]);
                at_mma(s[nt], ql[half * 2 + 1], b[2], b[3]);
                at_mma(s[nt], q[half * 2 + 1], bl[2], bl[3]);
            }
            at_mmap<QP>(s[nt], q[half * 2 + 0], b[0], b[1]);
            at_mmap<QP>(s[nt], q[half * 2 + 1], b[2], b[3]);
        }
    }
    float mx_a = m_a, mx_b = m_b;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            if (MASK) {
                int j = j0 + nt * 8 + (lane & 3) * 2 + e;
                if (j < lo_a || j > hi_a) s[nt][e] = -INFINITY;
                if (j < lo_b || j > hi_b) s[nt][2 + e] = -INFINITY;
            }
            mx_a = fmaxf(mx_a, s[nt][e]);
            mx_b = fmaxf(mx_b, s[nt][2 + e]);
        }
    mx_a = fmaxf(mx_a, at_shfl_xor(mx_a, 1));
    mx_a = fmaxf(mx_a, at_shfl_xor(mx_a, 2));
    mx_b = fmaxf(mx_b, at_shfl_xor(mx_b, 1));
    mx_b = fmaxf(mx_b, at_shfl_xor(mx_b, 2));
    /* m starts at the (finite) sink, so mx is finite and masked keys give 0 */
    const float al_a = at_ex2((m_a - mx_a) * AT_L2E), al_b = at_ex2((m_b - mx_b) * AT_L2E);
    const float c_a = mx_a * AT_L2E, c_b = mx_b * AT_L2E;
    m_a = mx_a; m_b = mx_b;
    float ps_a = 0.f, ps_b = 0.f;
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            s[nt][e] = at_ex2(fmaf(s[nt][e], AT_L2E, -c_a));
            s[nt][2 + e] = at_ex2(fmaf(s[nt][2 + e], AT_L2E, -c_b));
            ps_a += s[nt][e];
            ps_b += s[nt][2 + e];
        }
    l_a = l_a * al_a + ps_a;
    l_b = l_b * al_b + ps_b;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        o[nt][0] *= al_a; o[nt][1] *= al_a;
        o[nt][2] *= al_b; o[nt][3] *= al_b;
    }
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        const float *s0 = s[2 * kk], *s1 = s[2 * kk + 1];
        const uint32_t a[4] = {at_cvt<VP>(s0[0], s0[1]), at_cvt<VP>(s0[2], s0[3]),
                               at_cvt<VP>(s1[0], s1[1]), at_cvt<VP>(s1[2], s1[3])};
        uint32_t al[4];
        if (VP == AT_SPLIT) {
            al[0] = at_res(s0[0], s0[1], a[0]); al[1] = at_res(s0[2], s0[3], a[1]);
            al[2] = at_res(s1[0], s1[1], a[2]); al[3] = at_res(s1[2], s1[3], a[3]);
        }
#pragma unroll
        for (int dp = 0; dp < 4; dp++) {
            uint32_t b[4];
            const int off = at_swz(rbase + kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, dp * 2 + (lane >> 4));
            at_ldsm4t(b, Vs + off);
            if (VP == AT_SPLIT) {
                uint32_t bl[4];
                at_ldsm4t(bl, Vs + PL + off);
                at_mma(o[2 * dp], al, b[0], b[1]);
                at_mma(o[2 * dp], a, bl[0], bl[1]);
                at_mma(o[2 * dp + 1], al, b[2], b[3]);
                at_mma(o[2 * dp + 1], a, bl[2], bl[3]);
            }
            at_mmap<VP>(o[2 * dp], a, b[0], b[1]);
            at_mmap<VP>(o[2 * dp + 1], a, b[2], b[3]);
        }
    }
}

/* MUT=true is a deliberately wrong kernel (every window one key too long at
 * the top) used only to prove the benchmark's checker is sensitive. */
template <typename In, bool MUT, int QP, int VP>
__global__ __launch_bounds__(AT_THREADS, (AtLay<QP, VP>::MINB)) void k_attn_tc(
    const In *__restrict__ qkv, const int *__restrict__ sst, const int *__restrict__ sen,
    const float *__restrict__ sinks, uint16_t *__restrict__ ao, int T) {
    typedef AtLay<QP, VP> L;
    AT_DYN_SMEM(at_sm);
    unsigned char *Ks = at_sm + L::K, *Vs = at_sm + L::V;
    int *lo_s = (int *)(at_sm + L::LO), *hi_s = (int *)(at_sm + L::HI), *kr = (int *)(at_sm + L::KR);
    const int t0 = blockIdx.x * AT_Q, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    /* per-row windows; rows past T get an empty window (lo > every key) */
    if (tid < AT_Q) {
        int t = t0 + tid, lo = INT_MAX, hi = -1;
        if (t < T) {
            lo = max(t - AT_BAND, sst[t]);
            hi = min(t + AT_BAND, sen[t] - 1);
            if (MUT) hi = min(hi + 1, T - 1);
        }
        lo_s[tid] = lo; hi_s[tid] = hi;
        int a = at_rmin(lo), b = at_rmax(hi);
        if (lane == 0) { kr[warp * 2] = a; kr[warp * 2 + 1] = b; }
    }
    AT_SYNC();
    const int kmin = min(kr[0], kr[2]), kmax = max(kr[1], kr[3]);
    const int nk = kmax - kmin + 1, ns = min(nk + AT_KS, AT_ROWS);

    /* stage K/V rows [kmin, kmin+ns) of group g; rows >= nk are zero (only
     * ever read under a mask, and must not hold NaN/Inf) */
    for (int i = tid; i < ns * 16; i += AT_THREADS) {
        int r = i >> 4, part = i & 15, ch = part & 7;
        uint4 w = make_uint4(0, 0, 0, 0), wl = make_uint4(0, 0, 0, 0);
        const In *src = qkv + (size_t)(kmin + r) * AT_QKVD + AT_QD + (part >> 3) * AT_KVD + g * AT_HD + ch * 8;
        if (part < 8) {
            if (r < nk) at_ld8<QP>(src, w, wl);
            *(uint4 *)(Ks + at_swz(r, ch)) = w;
            if (QP == AT_SPLIT) *(uint4 *)(at_sm + L::KL + at_swz(r, ch)) = wl;
        } else {
            if (r < nk) at_ld8<VP>(src, w, wl);
            *(uint4 *)(Vs + at_swz(r, ch)) = w;
            if (VP == AT_SPLIT) *(uint4 *)(at_sm + L::VL + at_swz(r, ch)) = wl;
        }
    }
    AT_SYNC();

    const int h = g * AT_HPG + warp;
    const float sink = sinks[h] * AT_LN2;
    const int tig = lane & 3;
    for (int rg = 0; rg < AT_Q / 16; rg++) {
        const int ra = rg * 16 + (lane >> 2), rb = ra + 8, ta = t0 + ra, tb = t0 + rb;
        const int lo_a = lo_s[ra], hi_a = hi_s[ra], lo_b = lo_s[rb], hi_b = hi_s[rb];
        const int jlo = at_rmin(min(lo_a, lo_b)), jhi = at_rmax(max(hi_a, hi_b));
        if (jhi < jlo) continue; /* all 16 rows past T (warp-uniform) */
        const int ilo = at_rmax(max(lo_a, lo_b)), ihi = at_rmin(min(hi_a, hi_b));

        uint32_t q[4][4], ql[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int c = h * AT_HD + kk * 16 + tig * 2;
            const In *pa = qkv + (size_t)ta * AT_QKVD + c, *pb = qkv + (size_t)tb * AT_QKVD + c;
            q[kk][0] = q[kk][1] = q[kk][2] = q[kk][3] = 0u;
            ql[kk][0] = ql[kk][1] = ql[kk][2] = ql[kk][3] = 0u;
            if (ta < T) { at_ld2<QP>(pa, q[kk][0], ql[kk][0]); at_ld2<QP>(pa + 8, q[kk][2], ql[kk][2]); }
            if (tb < T) { at_ld2<QP>(pb, q[kk][1], ql[kk][1]); at_ld2<QP>(pb + 8, q[kk][3], ql[kk][3]); }
        }
        float o[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
        /* sink seeds the softmax: m = sink, l = exp(0) = 1 (counted once per quad) */
        float m_a = sink, m_b = sink, l_a = tig == 0 ? 1.f : 0.f, l_b = l_a;
        for (int j0 = jlo; j0 <= jhi; j0 += AT_KS) {
            if (j0 >= ilo && j0 + AT_KS - 1 <= ihi)
                at_step<false, QP, VP>(Ks, Vs, j0 - kmin, j0, q, ql, o, m_a, m_b, l_a, l_b, lo_a, hi_a, lo_b, hi_b, lane);
            else
                at_step<true, QP, VP>(Ks, Vs, j0 - kmin, j0, q, ql, o, m_a, m_b, l_a, l_b, lo_a, hi_a, lo_b, hi_b, lane);
        }
        l_a += at_shfl_xor(l_a, 1); l_a += at_shfl_xor(l_a, 2);
        l_b += at_shfl_xor(l_b, 1); l_b += at_shfl_xor(l_b, 2);
        const float ia = 1.f / l_a, ib = 1.f / l_b;
        if (ta < T) {
            uint16_t *dst = ao + (size_t)ta * AT_QD + h * AT_HD + tig * 2;
#pragma unroll
            for (int nt = 0; nt < 8; nt++) *(uint32_t *)(dst + nt * 8) = at_pack(o[nt][0] * ia, o[nt][1] * ia);
        }
        if (tb < T) {
            uint16_t *dst = ao + (size_t)tb * AT_QD + h * AT_HD + tig * 2;
#pragma unroll
            for (int nt = 0; nt < 8; nt++) *(uint32_t *)(dst + nt * 8) = at_pack(o[nt][2] * ib, o[nt][3] * ib);
        }
    }
}

#ifndef ATTN_EMU
template <typename In, bool MUT, int QP, int VP>
static void attn_launch(const In *qkv, const int *sst, const int *sen, const float *sinks, uint16_t *ao, int T,
                        cudaStream_t st) {
    constexpr int SMEM = AtLay<QP, VP>::END;
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(k_attn_tc<In, MUT, QP, VP>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        init = true;
    }
    if (T <= 0) return;
    k_attn_tc<In, MUT, QP, VP><<<dim3((T + AT_Q - 1) / AT_Q, 2), AT_THREADS, SMEM, st>>>(qkv, sst, sen, sinks, ao, T);
}
#ifndef ATTN_NO_API     /* bench_attn.cu instantiates every variant itself */
void attn_fwd_f32(const float *qkv, const int *sst, const int *sen, const float *sinks, uint16_t *ao, int T,
                  cudaStream_t st) {
    attn_launch<float, false, at_qp(ATTN_PREC), at_vp(ATTN_PREC)>(qkv, sst, sen, sinks, ao, T, st);
}
void attn_fwd_bf16(const uint16_t *qkv, const int *sst, const int *sen, const float *sinks, uint16_t *ao, int T,
                   cudaStream_t st) {
    attn_launch<uint16_t, false, AT_BF, AT_BF>(qkv, sst, sen, sinks, ao, T, st);
}
#endif
#endif
