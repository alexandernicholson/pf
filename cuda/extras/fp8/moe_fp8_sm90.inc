/* Opt-in FP8 (e4m3) expert GEMMs for sm_90a (PF_FP8_EXPERTS=1). Included inside namespace moe by moe.cu,
 * after moe_sm90.inc; the bf16 kernels above are untouched.
 *
 * Same CTA shape, routing, tiles and smem layout as k_wg: K-major SW128, 128-byte rows. An fp8 row of
 * 128 bytes holds 128 K elements, so a k-tile is BK = 128 (KT = 5) and each k32 wgmma advances the
 * descriptors by 32 bytes, exactly as the bf16 k16 one does. The accumulator fragment layout is the
 * same for every input type, so the epilogues only add the dequantization scales.
 *
 * Quantization (all scales fp32, values rounded RNE with satfinite to e4m3, max 448):
 *   w1q[e][n][k] = e4m3(w1[e][k][n] / s1[e][n]), s1 = amax_k |w1[e][k][n]| / 448   (per expert, per column)
 *   w2q[e][n][k] = e4m3(w2[e][k][n] / s2[e][n]), s2 likewise; stored as s2 / H_SCALE
 *   xq[t][k]     = e4m3(xh[t][k] / sx[t]),       sx = amax_k |xh[t][k]| / 448     (per token, dynamic)
 *   hq[r][i]     = e4m3(h[r][i] * H_SCALE)       static: |swiglu| < 7 * (7 + 1) = 56 -> 56 * 8 = 448
 * GEMM1 epilogue: h1 = acc * (sx * s1) + b1 -> SwiGLU -> hq.  GEMM2: out = (acc * s2' + b2) * gate. */

#include "wgmma_fp8.inc"                 /* cuda_fp8.h is included by moe.cu, outside the namespace */

constexpr float H_SCALE = 8.0f;

typedef uint8_t f8;

template <int NW> __device__ __forceinline__ void wgmma8_n(float *d, uint64_t da, uint64_t db);
template <> __device__ __forceinline__ void wgmma8_n<256>(float *d, uint64_t da, uint64_t db) { wgmma_e4m3_n256(d, da, db); }
template <> __device__ __forceinline__ void wgmma8_n<160>(float *d, uint64_t da, uint64_t db) { wgmma_e4m3_n160(d, da, db); }
template <> __device__ __forceinline__ void wgmma8_n<128>(float *d, uint64_t da, uint64_t db) { wgmma_e4m3_n128(d, da, db); }

__device__ __forceinline__ uint16_t f2e4m3x2(float lo, float hi) {
    return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}

/* warp per row of 640: q = e4m3(x / s), s = amax / 448; sc[row] = s * post (0 if the row is all zero) */
__global__ void k_quant_rows(const bf16 *__restrict__ x, int rows, f8 *__restrict__ q, float *__restrict__ sc,
                             float post) {
    int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const __nv_bfloat162 *src = (const __nv_bfloat162 *)(x + (size_t)r * 640);
    float2 v[10];
    float am = 0.0f;
#pragma unroll
    for (int j = 0; j < 10; j++) {
        v[j] = __bfloat1622float2(src[lane + 32 * j]);
        am = fmaxf(am, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    float s = am / 448.0f;
    uint16_t *dst = (uint16_t *)(q + (size_t)r * 640);
#pragma unroll
    for (int j = 0; j < 10; j++)
        dst[lane + 32 * j] = am > 0.0f ? f2e4m3x2(v[j].x / s, v[j].y / s) : (uint16_t)0;
    if (lane == 0) sc[r] = s * post;
}

template <int MODE, int BN, int NSUB, int ST, int PD = 1>
struct WG8 {
    static constexpr int BM = 128, BK = 128, NT = 256, K = 640, KT = K / BK, NW = BN / NSUB;
    static constexpr int ACC = BN / 2;
    static constexpr int ACH = BM * 128 / 16 / NT, BCH = BN * 128 / 16 / NT;   /* 16-byte chunks per thread */
    static constexpr size_t A_BYTES = BM * 128, B_BYTES = (size_t)BN * 128;
    static constexpr size_t SMEM = ST * (A_BYTES + B_BYTES) + 1024;
    static constexpr int GRID_X = MODE == 1 ? FF / (BN / 2) : D / BN;
    static_assert(MODE == 2 || NSUB == 1, "glu/lin pairing assumes one wgmma per k-step");
    static_assert(MODE == 1 ? FF % (BN / 2) == 0 : D % BN == 0, "grid covers the outputs");
    static constexpr int MINB = SMEM <= 113 * 1024 ? 2 : 1;
    static_assert((NW * 128) % 1024 == 0 && BN * 128 / 16 % NT == 0 && ST - 1 - PD >= 1, "shape");
};

/* A: MODE 1 xq [T][640] gathered through perm, MODE 2 hq [rows][640]; Wt: w1q / w2q (K-major fp8);
 * ws: s1 [e][1280] / s2' [e][640]; xs: sx [T] (MODE 1) */
template <int MODE, int BN, int NSUB, int ST, int PD>
__global__ void __launch_bounds__(256, (WG8<MODE, BN, NSUB, ST, PD>::MINB))
k_wg8(const f8 *__restrict__ A, const float *__restrict__ xs, const int *__restrict__ perm,
      const int4 *__restrict__ tiles, const int *__restrict__ ntiles, const f8 *__restrict__ Wt,
      const float *__restrict__ wsc, const bf16 *__restrict__ bias, const float *__restrict__ gates,
      f8 *__restrict__ Hout, float *__restrict__ out) {
    typedef WG8<MODE, BN, NSUB, ST, PD> G;
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int4 ti = tiles[tile];
    const int e = ti.x, row0 = ti.y;

    extern __shared__ uint8_t smem_raw[];
    const unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem_raw) + 1023) & ~1023u;
    const unsigned sA = sbase, sB = sbase + ST * G::A_BYTES;

    const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, wq = (tid >> 5) & 3;

    const int c8 = tid & 7, r8 = tid >> 3;
    const f8 *asrc[G::ACH]; bool aval[G::ACH];
#pragma unroll
    for (int i = 0; i < G::ACH; i++) {
        int r = r8 + 32 * i;
        if (MODE == 1) {
            int p = perm[row0 + r];
            aval[i] = p >= 0;
            asrc[i] = A + (size_t)(p >= 0 ? p / TOPK : 0) * G::K + c8 * 16;
        } else {
            aval[i] = true;
            asrc[i] = A + (size_t)(row0 + r) * G::K + c8 * 16;
        }
    }
    const int nloc = r8, sw = ((c8 ^ (r8 & 7)) * 16);
    const f8 *wbase = Wt + (size_t)e * (MODE == 1 ? 2 * FF : D) * G::K + c8 * 16;
    auto wrow = [&](int n) -> int {
        if (MODE == 1) return n < BN / 2 ? blockIdx.x * (BN / 2) + n : FF + blockIdx.x * (BN / 2) + n - BN / 2;
        return blockIdx.x * BN + n;
    };

    auto load = [&](int s, int kt) {
        int k0 = kt * G::BK;
        unsigned a = sA + s * G::A_BYTES, b = sB + s * G::B_BYTES;
#pragma unroll
        for (int i = 0; i < G::ACH; i++) {
            unsigned dst = a + (r8 + 32 * i) * 128 + sw;
            int sz = aval[i] ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(asrc[i] + k0), "r"(sz));
        }
#pragma unroll
        for (int i = 0; i < G::BCH; i++) {
            int n = nloc + 32 * i;
            unsigned dst = b + n * 128 + sw;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                         "l"(wbase + (size_t)wrow(n) * G::K + k0));
        }
    };

    float acc[G::ACC];
#pragma unroll
    for (int i = 0; i < G::ACC; i++) acc[i] = 0.0f;

#pragma unroll
    for (int s = 0; s < ST - 1 - PD; s++) { load(s, s); cp_commit(); }

#pragma unroll 1
    for (int kt = 0; kt < G::KT; kt++) {
        cp_wait<ST - 2 - PD>();
        fence_async_smem();
        __syncthreads();
        if (kt + ST - 1 - PD < G::KT) load((kt + ST - 1 - PD) % ST, kt + ST - 1 - PD);
        cp_commit();
        const int s = kt % ST;
        const uint64_t da = sw128_desc(sA + s * G::A_BYTES + wg * 64 * 128);
        const uint64_t db = sw128_desc(sB + s * G::B_BYTES);
#pragma unroll
        for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
        wg_fence();
#pragma unroll
        for (int k = 0; k < G::BK / 32; k++)
#pragma unroll
            for (int sub = 0; sub < NSUB; sub++)
                wgmma8_n<G::NW>(acc + sub * (G::NW / 2), da + 2 * k, db + 2 * k + sub * (G::NW * 128 >> 4));
        wg_commit();
        wg_wait<PD>();
#pragma unroll
        for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);

    const int rbase = row0 + wg * 64 + wq * 16 + (lane >> 2);
    if (MODE == 1) {
        const bf16 *b1 = bias + (size_t)e * 2 * FF;
        const float *s1 = wsc + (size_t)e * 2 * FF;
        float sx[2];
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            int p = perm[rbase + hf * 8];
            sx[hf] = p >= 0 ? xs[p / TOPK] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < BN / 16; j++) {
            int hc = blockIdx.x * (BN / 2) + j * 8 + (lane & 3) * 2;
            float bg0 = __bfloat162float(b1[hc]), bg1 = __bfloat162float(b1[hc + 1]);
            float bl0 = __bfloat162float(b1[FF + hc]), bl1 = __bfloat162float(b1[FF + hc + 1]);
            float sg0 = s1[hc], sg1 = s1[hc + 1], sl0 = s1[FF + hc], sl1 = s1[FF + hc + 1];
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const float *g = acc + j * 4 + hf * 2, *l = acc + (j + BN / 16) * 4 + hf * 2;
                float x = sx[hf];
                float h0 = swiglu(g[0] * (x * sg0) + bg0, l[0] * (x * sl0) + bl0);
                float h1 = swiglu(g[1] * (x * sg1) + bg1, l[1] * (x * sl1) + bl1);
                *(uint16_t *)(Hout + (size_t)(rbase + hf * 8) * FF + hc) = f2e4m3x2(h0 * H_SCALE, h1 * H_SCALE);
            }
        }
    } else {
        const bf16 *b2 = bias + (size_t)e * D;
        const float *s2 = wsc + (size_t)e * D;
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            int p = perm[rbase + hf * 8];
            if (p < 0) continue;
            float gw = gates[p];
#pragma unroll
            for (int sub = 0; sub < NSUB; sub++)
#pragma unroll
                for (int j = 0; j < G::NW / 8; j++) {
                    int col = blockIdx.x * BN + sub * G::NW + j * 8 + (lane & 3) * 2;
                    const float *a = acc + sub * (G::NW / 2) + j * 4 + hf * 2;
                    float2 v;
                    v.x = (a[0] * s2[col] + __bfloat162float(b2[col])) * gw;
                    v.y = (a[1] * s2[col + 1] + __bfloat162float(b2[col + 1])) * gw;
                    *(float2 *)(out + (size_t)p * D + col) = v;
                }
        }
    }
}

/* fp8 workspace = the bf16 one (H's bytes hold hq) + xq [T][640] + sx [T] */
struct WS8 { WS w; f8 *xq, *H8; float *xs; };

static size_t carve8(int T, void *base, WS8 *w) {
    size_t off = carve(T, base, w ? &w->w : nullptr);
    uint8_t *b = (uint8_t *)base;
    if (w) { w->xq = (f8 *)(b + off); w->H8 = (f8 *)w->w.H; }
    off += al((size_t)T * D);
    if (w) w->xs = (float *)(b + off);
    off += al((size_t)T * sizeof(float));
    return off;
}

template <int MODE, int BN, int NSUB, int ST, int PD>
static void launch_wg8(const WS8 &w, const f8 *A, const f8 *Wt, const float *wsc, const bf16 *bias,
                       const float *gates, float *out, cudaStream_t st) {
    typedef WG8<MODE, BN, NSUB, ST, PD> G;
    auto kfn = k_wg8<MODE, BN, NSUB, ST, PD>;
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)G::SMEM);
        init = true;
    }
    kfn<<<dim3(G::GRID_X, w.w.max_tiles), G::NT, G::SMEM, st>>>(A, w.xs, w.w.perm, w.w.tiles, w.w.ntiles, Wt, wsc,
                                                                 bias, gates, w.H8, out);
}

typedef void (*Gemm8Fn)(const WS8 &, const f8 *, const f8 *, const float *, const bf16 *, const float *, float *,
                        cudaStream_t);

template <int BN, int NSUB, int ST, int PD = 1>
struct CfgWG8 {
    template <int MODE>
    static void run(const WS8 &w, const f8 *A, const f8 *Wt, const float *wsc, const bf16 *bias, const float *gates,
                    float *out, cudaStream_t st) {
        launch_wg8<MODE, BN, NSUB, ST, PD>(w, A, Wt, wsc, bias, gates, out, st);
    }
};

/* runtime-selectable tile configs (moe_fp8_set_cfg / PF_FP8_CFG); name = <BN>s<stages>p<in-flight groups> */
struct Cfg8Entry { const char *name; Gemm8Fn g1, g2; };
#define CFG8(n, BN, NSUB, ST, PD) { n, CfgWG8<BN, NSUB, ST, PD>::run<1>, CfgWG8<BN, NSUB, ST, PD>::run<2> }
static const Cfg8Entry cfg8_table[] = {
    CFG8("160s3p0", 160, 1, 3, 0),
    CFG8("160s4p0", 160, 1, 4, 0),
    CFG8("160s4p1", 160, 1, 4, 1),
    CFG8("128s3p0", 128, 1, 3, 0),
    CFG8("128s4p1", 128, 1, 4, 1),
    { "256s3p0", CfgWG8<256, 1, 3, 0>::run<1>, nullptr },     /* GEMM1 only: 640 % 256 != 0 */
    { "320x2s3p0", nullptr, CfgWG8<320, 2, 3, 0>::run<2> },   /* GEMM2 only: 2 x n160 */
};
#undef CFG8
constexpr int NCFG8 = sizeof(cfg8_table) / sizeof(cfg8_table[0]);
static int cfg8_g1 = 0, cfg8_g2 = 0;

static void forward8(const bf16 *xh, const int *eidx, const float *gates, int T, const f8 *w1q, const float *s1,
                     const bf16 *b1, const f8 *w2q, const float *s2, const bf16 *b2, float *out, void *ws,
                     cudaStream_t st, Gemm8Fn g1, Gemm8Fn g2) {
    WS8 w;
    carve8(T, ws, &w);
    route(w.w, eidx, T, out, st);
    if (T > 0) k_quant_rows<<<(T + 7) / 8, 256, 0, st>>>(xh, T, w.xq, w.xs, 1.0f);
    g1(w, w.xq, w1q, s1, b1, gates, out, st);
    g2(w, w.H8, w2q, s2, b2, gates, out, st);
}
