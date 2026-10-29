/* Hopper (sm_90a) grouped GEMM: cp.async (gather) -> 128B-swizzled smem -> wgmma m64nNk16.
 * Included inside namespace moe by moe.cu.
 *
 * Weights are pre-transposed once to K-major (wt[e][n][k], see k_transpose) so A and B share
 * the canonical K-major SW128 layout: 128-byte rows of 64 bf16, 16-byte chunk c of row r stored
 * at chunk c ^ (r & 7), 8-row atoms 1024 B apart (SBO = 1024). Stage buffers are 1024-aligned.
 * CTA = 2 warpgroups x 64 rows = 128 rows (= PAD) by BN columns, BN = NSUB wgmma of width BN/NSUB.
 * MODE 1: BN = 256 -> 128 glu rows of w1t followed by the matching 128 lin rows; the epilogue
 *         pairs accumulator block j with j + 16 (same thread, same position).
 * MODE 2: BN = 320 (2 x n160) -> 2 CTAs cover the 640 outputs.
 * Pipeline: ST stages, PD wgmma groups left in flight, prefetch distance ST-1-PD. */

#include "wgmma.inc"

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void reg_fence(float &x) { asm volatile("" : "+f"(x)::"memory"); }

__device__ __forceinline__ uint64_t sw128_desc(unsigned saddr) {
    return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
           ((uint64_t)1 << 62);
}

template <int NW> __device__ __forceinline__ void wgmma_n(float *d, uint64_t da, uint64_t db);
template <> __device__ __forceinline__ void wgmma_n<256>(float *d, uint64_t da, uint64_t db) { wgmma_n256(d, da, db); }
template <> __device__ __forceinline__ void wgmma_n<160>(float *d, uint64_t da, uint64_t db) { wgmma_n160(d, da, db); }
template <> __device__ __forceinline__ void wgmma_n<128>(float *d, uint64_t da, uint64_t db) { wgmma_n128(d, da, db); }

template <int MODE, int BN, int NSUB, int ST, int PD = 1>
struct WG {
    static constexpr int BM = 128, BK = 64, NT = 256, K = 640, KT = K / BK, NW = BN / NSUB;
    static constexpr int ACC = BN / 2;                   /* fp32 accumulators per thread */
    static constexpr int ACH = BM * BK / 8 / NT, BCH = BN * BK / 8 / NT;
    static constexpr size_t A_BYTES = BM * BK * 2, B_BYTES = (size_t)BN * BK * 2;
    static constexpr size_t SMEM = ST * (A_BYTES + B_BYTES) + 1024;
    static constexpr int GRID_X = MODE == 1 ? FF / (BN / 2) : D / BN;
    static_assert(MODE == 2 || NSUB == 1, "glu/lin pairing assumes one wgmma per k-step");
    static constexpr int MINB = SMEM <= 113 * 1024 ? 2 : 1;
    static_assert((NW * 128) % 1024 == 0 && BN * BK / 8 % NT == 0 && ST - 1 - PD >= 1, "shape");
};

template <int MODE, int BN, int NSUB, int ST, int PD>
__global__ void __launch_bounds__(256, (WG<MODE, BN, NSUB, ST, PD>::MINB))
k_wg(const bf16 *__restrict__ A, const int *__restrict__ perm, const int4 *__restrict__ tiles,
     const int *__restrict__ ntiles, const bf16 *__restrict__ Wt, const bf16 *__restrict__ bias,
     const float *__restrict__ gates, bf16 *__restrict__ Hout, float *__restrict__ out) {
    typedef WG<MODE, BN, NSUB, ST, PD> G;
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int4 ti = tiles[tile];
    const int e = ti.x, row0 = ti.y;

    extern __shared__ uint8_t smem_raw[];
    const unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem_raw) + 1023) & ~1023u;
    uint8_t *sgen = smem_raw + (sbase - (unsigned)__cvta_generic_to_shared(smem_raw));
    const unsigned sA = sbase, sB = sbase + ST * G::A_BYTES;

    const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, wq = (tid >> 5) & 3;

    /* A: thread loads chunk c = tid & 7 of rows (tid >> 3) + 32 i */
    const int c8 = tid & 7, r8 = tid >> 3;
    const bf16 *asrc[G::ACH]; bool aval[G::ACH];
#pragma unroll
    for (int i = 0; i < G::ACH; i++) {
        int r = r8 + 32 * i;
        if (MODE == 1) {
            int p = perm[row0 + r];
            aval[i] = p >= 0;
            asrc[i] = A + (size_t)(p >= 0 ? p / TOPK : 0) * G::K + c8 * 8;
        } else {
            aval[i] = true;
            asrc[i] = A + (size_t)(row0 + r) * G::K + c8 * 8;
        }
    }
    const int nloc = r8, sw = ((c8 ^ (r8 & 7)) * 16); /* row&7 is invariant under +32 i */
    const bf16 *wbase = Wt + (size_t)e * (MODE == 1 ? 2 * FF : D) * G::K + c8 * 8;
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
        for (int k = 0; k < G::BK / 16; k++)
#pragma unroll
            for (int sub = 0; sub < NSUB; sub++)
                wgmma_n<G::NW>(acc + sub * (G::NW / 2), da + 2 * k, db + 2 * k + sub * (G::NW * 128 >> 4));
        wg_commit();
        wg_wait<PD>();
#pragma unroll
        for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
    (void)sgen;

    const int rbase = row0 + wg * 64 + wq * 16 + (lane >> 2);
    if (MODE == 1) {
        const bf16 *b1 = bias + (size_t)e * 2 * FF;
#pragma unroll
        for (int j = 0; j < BN / 16; j++) {
            int hc = blockIdx.x * (BN / 2) + j * 8 + (lane & 3) * 2;
            float bg0 = __bfloat162float(b1[hc]), bg1 = __bfloat162float(b1[hc + 1]);
            float bl0 = __bfloat162float(b1[FF + hc]), bl1 = __bfloat162float(b1[FF + hc + 1]);
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const float *g = acc + j * 4 + hf * 2, *l = acc + (j + BN / 16) * 4 + hf * 2;
                float h0 = swiglu(g[0] + bg0, l[0] + bl0), h1 = swiglu(g[1] + bg1, l[1] + bl1);
                *(__nv_bfloat162 *)(Hout + (size_t)(rbase + hf * 8) * FF + hc) = __floats2bfloat162_rn(h0, h1);
            }
        }
    } else {
        const bf16 *b2 = bias + (size_t)e * D;
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
                    v.x = (a[0] + __bfloat162float(b2[col])) * gw;
                    v.y = (a[1] + __bfloat162float(b2[col + 1])) * gw;
                    *(float2 *)(out + (size_t)p * D + col) = v;
                }
        }
    }
}

/* w[e][k][n] -> wt[e][n][k] */
__global__ void k_transpose(const bf16 *__restrict__ w, bf16 *__restrict__ wt, int K, int N) {
    __shared__ bf16 t[32][33];
    const bf16 *src = w + (size_t)blockIdx.z * K * N;
    bf16 *dst = wt + (size_t)blockIdx.z * K * N;
    int n0 = blockIdx.x * 32, k0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) t[i][threadIdx.x] = src[(size_t)(k0 + i) * N + n0 + threadIdx.x];
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) dst[(size_t)(n0 + i) * K + k0 + threadIdx.x] = t[threadIdx.x][i];
}

template <int MODE, int BN, int NSUB, int ST, int PD>
static void launch_wg(const WS &w, const bf16 *A, const bf16 *Wt, const bf16 *bias, const float *gates,
                      float *out, cudaStream_t st) {
    typedef WG<MODE, BN, NSUB, ST, PD> G;
    auto kfn = k_wg<MODE, BN, NSUB, ST, PD>;
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)G::SMEM);
        init = true;
    }
    kfn<<<dim3(G::GRID_X, w.max_tiles), G::NT, G::SMEM, st>>>(A, w.perm, w.tiles, w.ntiles, Wt, bias, gates, w.H, out);
}

template <int BN, int NSUB, int ST, int PD = 1>
struct CfgWG {
    template <int MODE>
    static void run(const WS &w, const bf16 *A, const bf16 *W, const bf16 *bias, const float *gates, float *out,
                    cudaStream_t st) {
        launch_wg<MODE, BN, NSUB, ST, PD>(w, A, W, bias, gates, out, st);
    }
};
