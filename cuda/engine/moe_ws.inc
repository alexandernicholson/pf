/* Warp-specialized persistent grouped GEMM (sm_90a), included inside namespace moe after moe_sm90.inc.
 *
 * CTA = 3 warpgroups, 1 CTA/SM, grid = #SMs, static round-robin over work items
 * w = tile * NX + nblk (tile from the device-built list, nblk = BN-column block; x fastest as in k_wg).
 *   WG0       producer: setmaxnreg.dec; cp.async gather of A (zero-fill for padding rows) and B
 *             (cp.async, or TMA for B when BTMA) into an ST-deep ring of SW128 stages; completion is
 *             signalled with cp.async.mbarrier.arrive.noinc (+ expect_tx/complete_tx for TMA) on full[s].
 *   WG1, WG2  consumers (ping-pong): consumer c owns local items i with i % 2 == c, a whole 128 x BN
 *             tile each (2 x wgmma m64nBNk16 per k16), one wgmma group kept in flight; stage s is
 *             released on empty[s] once the wgmma reading it has retired. Mainloops are serialized by
 *             the turn[] barriers (item i starts after item i-1 has issued its last wgmma), so one
 *             consumer's epilogue overlaps the other's mainloop. The serialization is also what keeps
 *             every full[] wait at most one phase ahead (parity waits are only valid then).
 * Global pipeline iteration g = local_item * KT + kt; stage = g % ST, use n = g / ST, parity n & 1.
 * Same smem layout, descriptors, K order and epilogues as k_wg, so results are expected bit-identical
 * to the E3 kernel (per-element accumulation: 40 sequential k16 wgmma steps into a zeroed fp32 acc). */

#include <cudaTypedefs.h>

__device__ __forceinline__ void mbar_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}\n" ::"r"(bar),
                 "r"(parity)
                 : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_2d(unsigned dst, const void *tmap, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
template <int N> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }

#include "ws_sched.h" /* stage / parity / turn math, also used by tests/test_ws.cpp */

template <int MODE, int BN, int ST, bool BTMA>
struct WS_ {
    static constexpr int BM = 128, BK = 64, K = 640, KT = K / BK, NTH = 384;
    static constexpr int NX = MODE == 1 ? FF / (BN / 2) : D / BN;
    static constexpr int ACC = BN;                            /* 2 m64 blocks x BN/2 per thread */
    static constexpr size_t A_BYTES = BM * BK * 2, B_BYTES = (size_t)BN * BK * 2;
    static constexpr size_t BAR_OFF = ST * (A_BYTES + B_BYTES);
    static constexpr size_t SMEM = BAR_OFF + (2 * ST + 2) * 8 + 1024;
    static constexpr int ACH = BM * 8 / 128, BCH = BN * 8 / 128; /* 16-byte chunks per producer thread */
    static constexpr unsigned B_TX = (unsigned)B_BYTES;
    /* setmaxnreg moves registers within the CTA's launch allocation (384 x 168, see ptxas -v), so
     * PRODUCER_REGS * 128 + CONSUMER_REGS * 256 must not exceed it or setmaxnreg.inc never returns */
    static constexpr int PRODUCER_REGS = BTMA ? 40 : BN == 128 ? 72 : 56;
    static constexpr int CONSUMER_REGS = BTMA ? 232 : BN == 128 ? 216 : 224;
    static_assert((MODE == 1 ? FF % (BN / 2) : D % BN) == 0, "BN must tile the output");
    static_assert(BN % 16 == 0 && (BN / 2) % 8 == 0 && B_BYTES % 1024 == 0, "shape");
    static_assert(MODE == 2 || (((BN / 2) * 128) % 1024 == 0 && (BN / 2) % 16 == 0), "lin half of B must stay 1024-aligned");
    static_assert(SMEM <= 227 * 1024, "smem");
    static_assert(PRODUCER_REGS * 128 + CONSUMER_REGS * 256 <= 168 * 384, "register pool");
};

template <int NW> __device__ __forceinline__ void wgmma_ws(float *d, uint64_t da, uint64_t db) { wgmma_n<NW>(d, da, db); }

template <int MODE, int BN, int ST, bool BTMA>
__global__ void __launch_bounds__(384, 1)
k_ws(const bf16 *__restrict__ A, const int *__restrict__ perm, const int4 *__restrict__ tiles,
     const int *__restrict__ ntiles, const bf16 *__restrict__ Wt, const bf16 *__restrict__ bias,
     const float *__restrict__ gates, bf16 *__restrict__ Hout, float *__restrict__ out,
     const __grid_constant__ CUtensorMap tmap) {
    typedef WS_<MODE, BN, ST, BTMA> G;
    const WsSched sc{ST, G::KT};
    extern __shared__ uint8_t smem_raw[];
    const unsigned sbase = ((unsigned)__cvta_generic_to_shared(smem_raw) + 1023) & ~1023u;
    const unsigned sA = sbase, sB = sbase + ST * G::A_BYTES;
    const unsigned full = sbase + G::BAR_OFF, empty = full + ST * 8, turn = empty + ST * 8;
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127;

    if (tid == 0) {
        for (int s = 0; s < ST; s++) {
            mbar_init(full + s * 8, 128 + (BTMA ? 1 : 0));
            mbar_init(empty + s * 8, 128);
        }
        mbar_init(turn, 128);
        mbar_init(turn + 8, 128);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    const int nitems = *ntiles * G::NX;

    if (wg == 0) {
        /* ---------------- producer ---------------- */
        setmaxnreg_dec<G::PRODUCER_REGS>();
        const int c8 = t & 7, r8 = t >> 3, sw = (c8 ^ (r8 & 7)) * 16; /* row & 7 invariant under +16 i */
        int g = 0;
#pragma unroll 1
        for (int w = blockIdx.x; w < nitems; w += gridDim.x) {
            const int tile = w / G::NX, nb = w % G::NX;
            const int4 ti = tiles[tile];
            const int e = ti.x, row0 = ti.y;
            /* A row offsets (elements); -1 = padding row (zero-filled) */
            int aoff[G::ACH];
#pragma unroll
            for (int i = 0; i < G::ACH; i++) {
                int r = r8 + 16 * i;
                if (MODE == 1) {
                    int p = perm[row0 + r];
                    aoff[i] = p >= 0 ? (p / TOPK) * G::K : -1;
                } else {
                    aoff[i] = (row0 + r) * G::K;
                }
            }
            const int wrow0 = e * (MODE == 1 ? 2 * FF : D);  /* first weight row of expert e */
            const int nglu = MODE == 1 ? nb * (BN / 2) : nb * BN;
#pragma unroll 1
            for (int kt = 0; kt < G::KT; kt++, g++) {
                const int s = sc.stage(g);
                mbar_wait(empty + s * 8, sc.empty_parity(g));
                const int k0 = kt * G::BK;
                const unsigned a = sA + s * G::A_BYTES, b = sB + s * G::B_BYTES, fb = full + s * 8;
#pragma unroll
                for (int i = 0; i < G::ACH; i++) {
                    unsigned dst = a + (r8 + 16 * i) * 128 + sw;
                    const bf16 *src = A + (aoff[i] >= 0 ? aoff[i] : 0) + k0 + c8 * 8;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                                 "r"(aoff[i] >= 0 ? 16 : 0));
                }
                if (BTMA) {
                    if (t == 0) {
                        mbar_expect_tx(fb, G::B_TX);
                        if (MODE == 1) {
                            tma_load_2d(b, &tmap, k0, wrow0 + nglu, fb);
                            tma_load_2d(b + (BN / 2) * 128, &tmap, k0, wrow0 + FF + nglu, fb);
                        } else {
                            tma_load_2d(b, &tmap, k0, wrow0 + nglu, fb);
                        }
                    }
                } else {
                    /* smem row n = r8 + 16 i <- weight row nglu + n (glu half) or FF + nglu + n - BN/2 (lin);
                     * BN/2 is a multiple of 16, so the half is a compile-time function of i */
                    const bf16 *bsrc = Wt + (size_t)(wrow0 + nglu + r8) * G::K + k0 + c8 * 8;
#pragma unroll
                    for (int i = 0; i < G::BCH; i++) {
                        const int roff = MODE == 1 && 16 * i >= BN / 2 ? 16 * i + FF - BN / 2 : 16 * i;
                        unsigned dst = b + (r8 + 16 * i) * 128 + sw;
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                                     "l"(bsrc + (size_t)roff * G::K));
                    }
                }
                cp_async_mbar_arrive(fb);
            }
        }
        cp_wait<0>(); /* do not retire with copies in flight */
        return;
    }

    /* ---------------- consumers ---------------- */
    setmaxnreg_inc<G::CONSUMER_REGS>();
    const int c = wg - 1, lane = tid & 31, wq = (tid >> 5) & 3;
    float acc[G::ACC];
    int j = 0;
#pragma unroll 1
    for (int w = blockIdx.x + c * gridDim.x; w < nitems; w += 2 * gridDim.x, j++) {
        const int li = 2 * j + c; /* local item index */
        const int tile = w / G::NX, nb = w % G::NX;
        const int4 ti = tiles[tile];
        const int e = ti.x, row0 = ti.y;
#pragma unroll
        for (int i = 0; i < G::ACC; i++) acc[i] = 0.0f;
        if (WsSched::turn_waits(c, j)) mbar_wait(turn + c * 8, WsSched::turn_parity(c, j));
        int g = li * G::KT;
#pragma unroll 1
        for (int kt = 0; kt < G::KT; kt++, g++) {
            const int s = sc.stage(g);
            mbar_wait(full + s * 8, sc.full_parity(g));
            fence_async_smem(); /* cp.async (generic proxy) writes -> wgmma (async proxy) reads */
            const uint64_t da = sw128_desc(sA + s * G::A_BYTES), db = sw128_desc(sB + s * G::B_BYTES);
#pragma unroll
            for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
            wg_fence();
#pragma unroll
            for (int k = 0; k < G::BK / 16; k++) {
                wgmma_ws<BN>(acc, da + 2 * k, db + 2 * k);
                wgmma_ws<BN>(acc + BN / 2, da + 2 * k + (64 * 128 >> 4), db + 2 * k);
            }
            wg_commit();
            wg_wait<1>();
#pragma unroll
            for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
            if (kt > 0) mbar_arrive(empty + sc.stage(g - 1) * 8);
        }
        /* all wgmma of this item issued: hand the tensor cores to the other consumer */
        mbar_arrive(turn + (1 - c) * 8);
        wg_wait<0>();
#pragma unroll
        for (int i = 0; i < G::ACC; i++) reg_fence(acc[i]);
        mbar_arrive(empty + sc.stage(g - 1) * 8);

#pragma unroll
        for (int mb = 0; mb < 2; mb++) {
            const float *am = acc + mb * (BN / 2);
            const int rbase = row0 + mb * 64 + wq * 16 + (lane >> 2);
            if (MODE == 1) {
                const bf16 *b1 = bias + (size_t)e * 2 * FF;
#pragma unroll
                for (int jj = 0; jj < BN / 16; jj++) {
                    int hc = nb * (BN / 2) + jj * 8 + (lane & 3) * 2;
                    float bg0 = __bfloat162float(b1[hc]), bg1 = __bfloat162float(b1[hc + 1]);
                    float bl0 = __bfloat162float(b1[FF + hc]), bl1 = __bfloat162float(b1[FF + hc + 1]);
#pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        const float *gl = am + jj * 4 + hf * 2, *l = am + (jj + BN / 16) * 4 + hf * 2;
                        float h0 = swiglu(gl[0] + bg0, l[0] + bl0), h1 = swiglu(gl[1] + bg1, l[1] + bl1);
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
                    for (int jj = 0; jj < BN / 8; jj++) {
                        int col = nb * BN + jj * 8 + (lane & 3) * 2;
                        const float *a = am + jj * 4 + hf * 2;
                        float2 v;
                        v.x = (a[0] + __bfloat162float(b2[col])) * gw;
                        v.y = (a[1] + __bfloat162float(b2[col + 1])) * gw;
                        *(float2 *)(out + (size_t)p * D + col) = v;
                    }
                }
            }
        }
    }
}

/* TMA descriptor for the K-major weights viewed as 2-D [rows][640] bf16, box {64, box_rows}, SW128.
 * Encoded on the host through the runtime's driver entry point (no -lcuda); cached per pointer. */
static bool make_tmap(CUtensorMap *m, const bf16 *wt, int rows, int box_rows) {
    static PFN_cuTensorMapEncodeTiled_v12000 enc = nullptr;
    if (!enc) {
        cudaDriverEntryPointQueryResult q;
        void *fn = nullptr;
        if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess || !fn)
            return false;
        enc = (PFN_cuTensorMapEncodeTiled_v12000)fn;
    }
    cuuint64_t dims[2] = {640, (cuuint64_t)rows};
    cuuint64_t strides[1] = {640 * sizeof(bf16)};
    cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
    cuuint32_t estr[2] = {1, 1};
    return enc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)wt, dims, strides, box, estr,
               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static int num_sms() {
    static int n = 0;
    if (!n) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0) n = 132;
    }
    return n;
}

template <int MODE, int BN, int ST, bool BTMA>
static void launch_ws(const WS &w, const bf16 *A, const bf16 *Wt, const bf16 *bias, const float *gates,
                      float *out, cudaStream_t st) {
    typedef WS_<MODE, BN, ST, BTMA> G;
    auto kfn = k_ws<MODE, BN, ST, BTMA>;
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)G::SMEM);
        init = true;
    }
    CUtensorMap tm;
    memset(&tm, 0, sizeof(tm));
    if (BTMA) {
        static const bf16 *cached_ptr = nullptr;
        static CUtensorMap cached;
        if (Wt != cached_ptr) {
            if (!make_tmap(&cached, Wt, NE * (MODE == 1 ? 2 * FF : D), MODE == 1 ? BN / 2 : BN)) {
                fprintf(stderr, "moe: cuTensorMapEncodeTiled failed\n");
                abort();
            }
            cached_ptr = Wt;
        }
        tm = cached;
    }
    kfn<<<num_sms(), G::NTH, G::SMEM, st>>>(A, w.perm, w.tiles, w.ntiles, Wt, bias, gates, w.H, out, tm);
}

template <int BN1, int ST1, int BN2, int ST2, bool BTMA>
struct CfgWS {
    template <int MODE>
    static void run(const WS &w, const bf16 *A, const bf16 *W, const bf16 *bias, const float *gates, float *out,
                    cudaStream_t st) {
        if (MODE == 1) launch_ws<1, BN1, ST1, BTMA>(w, A, W, bias, gates, out, st);
        else launch_ws<2, BN2, ST2, BTMA>(w, A, W, bias, gates, out, st);
    }
};

/* GEMM1 from one config, GEMM2 from another (the H interface between them is identical) */
template <class C1, class C2>
struct CfgMix {
    template <int MODE>
    static void run(const WS &w, const bf16 *A, const bf16 *W, const bf16 *bias, const float *gates, float *out,
                    cudaStream_t st) {
        if (MODE == 1) C1::template run<1>(w, A, W, bias, gates, out, st);
        else C2::template run<2>(w, A, W, bias, gates, out, st);
    }
};
