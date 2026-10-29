/* Fused grouped MoE expert kernels (mma.sync m16n8k16 bf16, ldmatrix, cp.async multistage).
 *
 * Pipeline (all on device, no host sync):
 *   k_hist     per-expert pair counts (smem histogram + one global atomic per block/expert)
 *   k_scan     segment offsets padded to PAD rows, tile list {e, row0, row_end}, padding perm = -1
 *   k_scatter  perm[sorted row] = pair index p = t*4+s (order within an expert is irrelevant:
 *              rows are computed independently, so results are bit-identical run to run)
 *   k_gemm<1>  gather xh rows on load, W1 tile = 64 glu + 64 lin columns interleaved per warp,
 *              epilogue bias + SwiGLU + bf16 -> H [sorted rows][640]
 *   k_gemm<2>  H @ W2, epilogue (acc + b2) * gate scattered to out[p][640]
 */
#include "moe.h"
#include <stdint.h>

namespace moe {

constexpr int D = 640, FF = 640, NE = 128, TOPK = 4;
constexpr int PAD = 128;               /* segment padding granularity = largest BM */
constexpr float SWIGLU_LIMIT = 7.0f, SWIGLU_ALPHA = 1.702f;

typedef __nv_bfloat16 bf16;

struct WS {
    int *counts, *cursor, *seg, *ntiles;
    int4 *tiles;
    int *perm;
    bf16 *H;
    int max_tiles;
};

static size_t al(size_t x) { return (x + 255) & ~(size_t)255; }

static int max_tiles(int T) { return (int)(((size_t)T * TOPK + (size_t)NE * (PAD - 1)) / PAD); }

static size_t carve(int T, void *base, WS *w) {
    int mt = max_tiles(T);
    size_t rows = (size_t)mt * PAD, off = 0;
    uint8_t *b = (uint8_t *)base;
#define TAKE(ptr, type, n) do { if (w) w->ptr = (type *)(b + off); off += al((size_t)(n) * sizeof(type)); } while (0)
    TAKE(counts, int, NE);
    TAKE(cursor, int, NE);
    TAKE(seg, int, NE + 1);
    TAKE(ntiles, int, 1);
    TAKE(tiles, int4, mt);
    TAKE(perm, int, rows);
    TAKE(H, bf16, rows * FF);
#undef TAKE
    if (w) w->max_tiles = mt;
    return off;
}

/* ---------------- routing ---------------- */

__global__ void k_hist(const int *__restrict__ eidx, int P, int *__restrict__ counts,
                       float *__restrict__ out) {
    __shared__ int sh[NE];
    for (int i = threadIdx.x; i < NE; i += blockDim.x) sh[i] = 0;
    __syncthreads();
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < P; p += gridDim.x * blockDim.x) {
        int e = eidx[p];
        if (e >= 0) atomicAdd(&sh[e], 1);
        else for (int j = 0; j < D; j++) out[(size_t)p * D + j] = 0.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NE; i += blockDim.x)
        if (sh[i]) atomicAdd(&counts[i], sh[i]);
}

__global__ void k_scan(const int *__restrict__ counts, int *__restrict__ seg, int *__restrict__ cursor,
                       int *__restrict__ ntiles, int4 *__restrict__ tiles, int *__restrict__ perm) {
    __shared__ int sp[NE], st[NE];
    int e = threadIdx.x;
    int c = counts[e], pc = (c + PAD - 1) / PAD * PAD, nt = pc / PAD;
    sp[e] = pc; st[e] = nt;
    __syncthreads();
    if (e == 0) {
        int a = 0, b = 0;
        for (int i = 0; i < NE; i++) {
            int x = sp[i], y = st[i];
            sp[i] = a; st[i] = b; a += x; b += y;
        }
        seg[NE] = a; *ntiles = b;
    }
    __syncthreads();
    int off = sp[e];
    seg[e] = off; cursor[e] = off;
    for (int j = 0; j < nt; j++) tiles[st[e] + j] = make_int4(e, off + j * PAD, off + c, 0);
    for (int i = off + c; i < off + pc; i++) perm[i] = -1;
}

__global__ void k_scatter(const int *__restrict__ eidx, int P, int *__restrict__ cursor,
                          int *__restrict__ perm) {
    __shared__ int sh[NE], base[NE];
    for (int i = threadIdx.x; i < NE; i += blockDim.x) sh[i] = 0;
    __syncthreads();
    int p = blockIdx.x * blockDim.x + threadIdx.x, e = -1, r = 0;
    if (p < P) { e = eidx[p]; if (e >= 0) r = atomicAdd(&sh[e], 1); }
    __syncthreads();
    for (int i = threadIdx.x; i < NE; i += blockDim.x)
        if (sh[i]) base[i] = atomicAdd(&cursor[i], sh[i]);
    __syncthreads();
    if (e >= 0) perm[base[e] + r] = p;
}

/* ---------------- tensor-core primitives ---------------- */

__device__ __forceinline__ void cp16(void *s, const void *g, bool v) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    int n = v ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t *r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t *r, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float *d, const uint32_t *a, const uint32_t *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float swiglu(float glu, float lin) {
    glu = fminf(glu, SWIGLU_LIMIT);
    lin = fminf(fmaxf(lin, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    return glu / (1.0f + expf(-SWIGLU_ALPHA * glu)) * (lin + 1.0f);
}

/* ---------------- grouped GEMM ----------------
 * MODE 1: A = gathered xh rows, W = w1[e] (640 x 1280), the CTA's BN columns are BN/2 glu +
 *         BN/2 lin columns arranged so each warp owns matching glu/lin halves.
 * MODE 2: A = H rows (contiguous), W = w2[e] (640 x 640).
 * smem rows are 128 B (A, BK=64) / BN*2 B (B); 16-byte chunks XOR-swizzled by row&7. */
template <int MODE, int BM, int BN, int BK, int ST, int WM, int WN>
struct Gemm {
    static constexpr int NT = WM * WN * 32, K = 640, KT = K / BK;
    static constexpr int LDW = MODE == 1 ? 2 * FF : D;
    static constexpr int WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;
    static constexpr int ACH = BM * BK / 8 / NT, BCH = BK * BN / 8 / NT;
    static constexpr int SPLIT = PAD / BM;
    static constexpr size_t SMEM = (size_t)ST * (BM * BK + BK * BN) * sizeof(bf16);
    static constexpr int GRID_X = MODE == 1 ? FF / (BN / 2) : D / BN;
    static_assert(BK == 64, "swizzle assumes 128-byte A rows");
    static_assert(BM * BK / 8 % NT == 0 && BK * BN / 8 % NT == 0, "load split");
    static_assert(NI % 2 == 0 && MI >= 1 && PAD % BM == 0, "tile shape");
    static_assert(MODE == 2 || (WTN / 2) % 8 == 0, "glu/lin halves");
};

template <int MODE, int BM, int BN, int BK, int ST, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32)
k_gemm(const bf16 *__restrict__ A, const int *__restrict__ perm, const int4 *__restrict__ tiles,
       const int *__restrict__ ntiles, const bf16 *__restrict__ W, const bf16 *__restrict__ bias,
       const float *__restrict__ gates, bf16 *__restrict__ Hout, float *__restrict__ out) {
    typedef Gemm<MODE, BM, BN, BK, ST, WM, WN> G;
    const int tile = blockIdx.y / G::SPLIT, sub = blockIdx.y % G::SPLIT;
    if (tile >= *ntiles) return;
    const int4 ti = tiles[tile];
    const int e = ti.x, row0 = ti.y + sub * BM, rowend = ti.z;
    if (row0 >= rowend) return;

    extern __shared__ __align__(128) uint8_t smem[];
    bf16 *sA = (bf16 *)smem, *sB = sA + ST * BM * BK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;

    const bf16 *asrc[G::ACH]; int adst[G::ACH]; bool aval[G::ACH];
#pragma unroll
    for (int i = 0; i < G::ACH; i++) {
        int idx = tid + i * G::NT, r = idx / (BK / 8), c = idx % (BK / 8);
        if (MODE == 1) {
            int p = perm[row0 + r];
            aval[i] = p >= 0;
            asrc[i] = A + (size_t)(p >= 0 ? p / TOPK : 0) * G::K + c * 8;
        } else {
            aval[i] = true;
            asrc[i] = A + (size_t)(row0 + r) * G::K + c * 8;
        }
        adst[i] = r * BK + ((c ^ (r & 7)) * 8);
    }
    const bf16 *wbase = W + (size_t)e * G::K * G::LDW;
    const bf16 *bsrc[G::BCH]; int bdst[G::BCH];
#pragma unroll
    for (int i = 0; i < G::BCH; i++) {
        int idx = tid + i * G::NT, k = idx / (BN / 8), c = idx % (BN / 8), col;
        if (MODE == 1) {
            int cc = c * 8, g = cc / G::WTN, w = cc % G::WTN, h = G::WTN / 2;
            int base = blockIdx.x * (BN / 2) + g * h;
            col = w < h ? base + w : FF + base + (w - h);
        } else {
            col = blockIdx.x * BN + c * 8;
        }
        bsrc[i] = wbase + (size_t)k * G::LDW + col;
        bdst[i] = k * BN + ((c ^ (k & 7)) * 8);
    }

    auto load = [&](int s, int kt) {
        int k0 = kt * BK;
        bf16 *a = sA + s * BM * BK, *b = sB + s * BK * BN;
#pragma unroll
        for (int i = 0; i < G::ACH; i++) cp16(a + adst[i], asrc[i] + k0, aval[i]);
#pragma unroll
        for (int i = 0; i < G::BCH; i++) cp16(b + bdst[i], bsrc[i] + (size_t)k0 * G::LDW, true);
    };

    float acc[G::MI][G::NI][4];
#pragma unroll
    for (int i = 0; i < G::MI; i++)
#pragma unroll
        for (int j = 0; j < G::NI; j++)
#pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.0f;

#pragma unroll
    for (int s = 0; s < ST - 1; s++) { load(s, s); cp_commit(); }

    const unsigned sAa = (unsigned)__cvta_generic_to_shared(sA);
    const unsigned sBa = (unsigned)__cvta_generic_to_shared(sB);

#pragma unroll 1
    for (int kt = 0; kt < G::KT; kt++) {
        cp_wait<ST - 2>();
        __syncthreads();
        if (kt + ST - 1 < G::KT) load((kt + ST - 1) % ST, kt + ST - 1);
        cp_commit();
        const int s = kt % ST;
        const unsigned a_s = sAa + s * BM * BK * 2, b_s = sBa + s * BK * BN * 2;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            uint32_t af[G::MI][4], bfr[G::NI][2];
#pragma unroll
            for (int mi = 0; mi < G::MI; mi++) {
                int r = wm * G::WTM + mi * 16 + (lane & 15), c = kk * 2 + (lane >> 4);
                ldsm4(af[mi], a_s + (r * BK + ((c ^ (r & 7)) * 8)) * 2);
            }
#pragma unroll
            for (int nj = 0; nj < G::NI / 2; nj++) {
                int k = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int c = (wn * G::WTN) / 8 + nj * 2 + (lane >> 4);
                uint32_t r4[4];
                ldsm4t(r4, b_s + (k * BN + ((c ^ (k & 7)) * 8)) * 2);
                bfr[2 * nj][0] = r4[0]; bfr[2 * nj][1] = r4[1];
                bfr[2 * nj + 1][0] = r4[2]; bfr[2 * nj + 1][1] = r4[3];
            }
#pragma unroll
            for (int mi = 0; mi < G::MI; mi++)
#pragma unroll
                for (int ni = 0; ni < G::NI; ni++) mma16816(acc[mi][ni], af[mi], bfr[ni]);
        }
    }
    cp_wait<0>();

    if (MODE == 1) {
        const bf16 *b1 = bias + (size_t)e * 2 * FF;
#pragma unroll
        for (int ni = 0; ni < G::NI / 2; ni++) {
            int hc = blockIdx.x * (BN / 2) + wn * (G::WTN / 2) + ni * 8 + (lane & 3) * 2;
            float bg0 = __bfloat162float(b1[hc]), bg1 = __bfloat162float(b1[hc + 1]);
            float bl0 = __bfloat162float(b1[FF + hc]), bl1 = __bfloat162float(b1[FF + hc + 1]);
#pragma unroll
            for (int mi = 0; mi < G::MI; mi++)
#pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    int row = row0 + wm * G::WTM + mi * 16 + (lane >> 2) + hf * 8;
                    const float *g = acc[mi][ni] + hf * 2, *l = acc[mi][ni + G::NI / 2] + hf * 2;
                    float h0 = swiglu(g[0] + bg0, l[0] + bl0), h1 = swiglu(g[1] + bg1, l[1] + bl1);
                    *(__nv_bfloat162 *)(Hout + (size_t)row * FF + hc) = __floats2bfloat162_rn(h0, h1);
                }
        }
    } else {
        const bf16 *b2 = bias + (size_t)e * D;
#pragma unroll
        for (int mi = 0; mi < G::MI; mi++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                int row = row0 + wm * G::WTM + mi * 16 + (lane >> 2) + hf * 8;
                int p = perm[row];
                if (p < 0) continue;
                float gw = gates[p];
#pragma unroll
                for (int ni = 0; ni < G::NI; ni++) {
                    int col = blockIdx.x * BN + wn * G::WTN + ni * 8 + (lane & 3) * 2;
                    float2 v;
                    v.x = (acc[mi][ni][hf * 2] + __bfloat162float(b2[col])) * gw;
                    v.y = (acc[mi][ni][hf * 2 + 1] + __bfloat162float(b2[col + 1])) * gw;
                    *(float2 *)(out + (size_t)p * D + col) = v;
                }
            }
    }
}

template <int MODE, int BM, int BN, int BK, int ST, int WM, int WN>
static void launch_gemm(const WS &w, const bf16 *A, const bf16 *W, const bf16 *bias, const float *gates,
                        float *out, cudaStream_t st) {
    typedef Gemm<MODE, BM, BN, BK, ST, WM, WN> G;
    auto kfn = k_gemm<MODE, BM, BN, BK, ST, WM, WN>;
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)G::SMEM);
        init = true;
    }
    dim3 grid(G::GRID_X, w.max_tiles * G::SPLIT);
    kfn<<<grid, G::NT, G::SMEM, st>>>(A, w.perm, w.tiles, w.ntiles, W, bias, gates, w.H, out);
}

static void route(const WS &w, const int *eidx, int T, float *out, cudaStream_t st) {
    int P = T * TOPK;
    cudaMemsetAsync(w.counts, 0, NE * sizeof(int), st);
    int hb = (P + 1023) / 1024; if (hb > 1024) hb = 1024; if (hb < 1) hb = 1;
    k_hist<<<hb, 256, 0, st>>>(eidx, P, w.counts, out);
    k_scan<<<1, NE, 0, st>>>(w.counts, w.seg, w.cursor, w.ntiles, w.tiles, w.perm);
    if (P > 0) k_scatter<<<(P + 1023) / 1024, 1024, 0, st>>>(eidx, P, w.cursor, w.perm);
}

/* tile configs: <BM, BN, BK, STAGES, WARPS_M, WARPS_N> */
template <class C1, class C2>
static void forward(const bf16 *xh, const int *eidx, const float *gates, int T, const bf16 *w1,
                    const bf16 *b1, const bf16 *w2, const bf16 *b2, float *out, void *ws, cudaStream_t st) {
    WS w;
    carve(T, ws, &w);
    route(w, eidx, T, out, st);
    C1::template run<1>(w, xh, w1, b1, gates, out, st);
    C2::template run<2>(w, w.H, w2, b2, gates, out, st);
}

template <int BM, int BN, int BK, int ST, int WM, int WN>
struct Cfg {
    template <int MODE>
    static void run(const WS &w, const bf16 *A, const bf16 *W, const bf16 *bias, const float *gates,
                    float *out, cudaStream_t st) {
        launch_gemm<MODE, BM, BN, BK, ST, WM, WN>(w, A, W, bias, gates, out, st);
    }
};

typedef Cfg<128, 128, 64, 3, 2, 4> Cfg1Default;
typedef Cfg<128, 128, 64, 3, 2, 4> Cfg2Default;

#ifndef MOE_NO_SM90
#include "moe_sm90.inc"
typedef CfgWG<160, 1, 3, 0> CfgWG1Default; /* 80 glu + 80 lin columns, 2 CTAs/SM */
typedef CfgWG<160, 1, 3, 0> CfgWG2Default;
#endif

} // namespace moe

extern "C" size_t moe_workspace_size(int T) { return moe::carve(T, nullptr, nullptr); }

extern "C" void moe_forward(const __nv_bfloat16 *xh, const int *eidx, const float *gates, int T,
                            const __nv_bfloat16 *w1, const __nv_bfloat16 *b1, const __nv_bfloat16 *w2,
                            const __nv_bfloat16 *b2, float *out, void *workspace, cudaStream_t stream) {
    moe::forward<moe::Cfg1Default, moe::Cfg2Default>(xh, eidx, gates, T, w1, b1, w2, b2, out, workspace,
                                                     stream);
}

#ifndef MOE_NO_SM90
extern "C" void moe_prepare_weights(const __nv_bfloat16 *w1, const __nv_bfloat16 *w2, __nv_bfloat16 *w1t,
                                    __nv_bfloat16 *w2t, cudaStream_t stream) {
    using namespace moe;
    k_transpose<<<dim3(2 * FF / 32, D / 32, NE), dim3(32, 8), 0, stream>>>(w1, w1t, D, 2 * FF);
    k_transpose<<<dim3(D / 32, FF / 32, NE), dim3(32, 8), 0, stream>>>(w2, w2t, FF, D);
}

extern "C" void moe_forward_sm90(const __nv_bfloat16 *xh, const int *eidx, const float *gates, int T,
                                 const __nv_bfloat16 *w1t, const __nv_bfloat16 *b1, const __nv_bfloat16 *w2t,
                                 const __nv_bfloat16 *b2, float *out, void *workspace, cudaStream_t stream) {
    moe::forward<moe::CfgWG1Default, moe::CfgWG2Default>(xh, eidx, gates, T, w1t, b1, w2t, b2, out, workspace,
                                                         stream);
}
#endif
