/* gemm_bi.cu — batch-invariant bf16 GEMM, fp32 out; see gemm_bi.h.
 *
 * CTA = 128 x 128 output tile, 8 warps as 2 (M) x 4 (N), warp tile 64 x 32
 * (4 x 4 m16n8 accumulators). K is swept in 32-wide stages through a 4-deep
 * cp.async ring (rows past M are zero-filled and never stored); each stage is
 * two k16 steps of ldmatrix.x4 + mma.sync. Shared rows are padded to 40
 * elements (80 B), which makes the 8-row ldmatrix phases conflict-free.
 *
 * Per output element the reduction is acc = 0, then one MMA per k16 block in
 * ascending K: nothing depends on M, blockIdx or the neighbouring rows.
 *
 * Built with -DGEMM_EMU the kernel compiles for the host emulator
 * (experiments/E13's emu_rt.h): cp.async becomes a plain copy.
 */
#include "gemm_bi.h"

#ifdef GEMM_EMU
#include "emu_rt.h"
#define GB_SYNC() AT_SYNC()
inline void gb_cp16(void *dst, const void *src, bool ok) {
    if (ok) memcpy(dst, src, 16); else memset(dst, 0, 16);
}
inline void gb_commit() {}
template <int N> inline void gb_wait() {}
#define gb_ldsm4 at_ldsm4
#define gb_mma at_mma
#else
#include <stdio.h>
#include <stdlib.h>
#define GB_SYNC() __syncthreads()
/* 16-byte async copy, zero-filled when !ok (src must still be a valid address) */
__device__ __forceinline__ void gb_cp16(void *dst, const void *src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void gb_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void gb_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void gb_ldsm4(uint32_t *r, const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void gb_mma(float *d, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif

#define GB_BM 128
#define GB_BN 128
#define GB_BK 32
#define GB_ST 4
#define GB_LD 40                                    /* padded smem row (elements) */
#define GB_THREADS 256
#define GB_STAGE ((GB_BM + GB_BN) * GB_LD)          /* elements per stage */
#define GB_SMEM (GB_ST * GB_STAGE * 2)              /* 81920 B: 2 CTAs per SM */

__global__ __launch_bounds__(GB_THREADS, 2) void k_gemm_bi(const uint16_t *__restrict__ X,
                                                          const uint16_t *__restrict__ W,
                                                          float *__restrict__ C, int M, int N, int K) {
#ifdef GEMM_EMU
    uint16_t *S = (uint16_t *)emu_blk->smem.data();
#else
    extern __shared__ __align__(128) unsigned char gb_sm[];
    uint16_t *S = (uint16_t *)gb_sm;
#endif
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * GB_BM, n0 = blockIdx.x * GB_BN, KT = K / GB_BK;

    /* stage kt -> ring slot: 128 rows x 4 16-byte chunks of X and of W, 2 + 2 per thread */
    auto load = [&](int kt, int slot) {
        uint16_t *As = S + slot * GB_STAGE, *Bs = As + GB_BM * GB_LD;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = tid + i * GB_THREADS, r = c >> 2, ch = c & 3, k = kt * GB_BK + ch * 8;
            const bool ok = m0 + r < M;
            gb_cp16(As + r * GB_LD + ch * 8, X + (size_t)(ok ? m0 + r : 0) * K + k, ok);
            gb_cp16(Bs + r * GB_LD + ch * 8, W + (size_t)(n0 + r) * K + k, true);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

#pragma unroll
    for (int s = 0; s < GB_ST - 1; s++) {
        if (s < KT) load(s, s);
        gb_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        gb_wait<GB_ST - 2>();          /* stage kt has landed (this thread's copies) */
        GB_SYNC();                     /* ... everyone's, and slot (kt-1) % ST is free */
        if (kt + GB_ST - 1 < KT) load(kt + GB_ST - 1, (kt + GB_ST - 1) % GB_ST);
        gb_commit();
        const uint16_t *As = S + (kt % GB_ST) * GB_STAGE, *Bs = As + GB_BM * GB_LD;
#pragma unroll
        for (int ks = 0; ks < GB_BK / 16; ks++) {
            uint32_t a[4][4], b[2][4];
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
                gb_ldsm4(a[mi], As + (wm * 64 + mi * 16 + (lane & 15)) * GB_LD + ks * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int nj = 0; nj < 2; nj++)
                gb_ldsm4(b[nj], Bs + (wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * GB_LD + ks * 16 +
                                    ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    gb_mma(acc[mi][ni], a[mi], b[ni >> 1][(ni & 1) * 2], b[ni >> 1][(ni & 1) * 2 + 1]);
        }
    }
    gb_wait<0>();

    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int row = m0 + wm * 64 + mi * 16 + g + hf * 8;
            if (row >= M) continue;
            float *dst = C + (size_t)row * N + n0 + wn * 32 + t4 * 2;
#pragma unroll
            for (int ni = 0; ni < 4; ni++)
                *(float2 *)(dst + ni * 8) = make_float2(acc[mi][ni][hf * 2], acc[mi][ni][hf * 2 + 1]);
        }
}

#ifdef GEMM_EMU
inline void gemm_bi_emu(const uint16_t *X, const uint16_t *W, float *C, int M, int N, int K) {
    emu_launch(N / GB_BN, (M + GB_BM - 1) / GB_BM, GB_THREADS, GB_SMEM, [&] { k_gemm_bi(X, W, C, M, N, K); });
}
#else
extern "C" void gemm_bi_nt(const uint16_t *X, const uint16_t *W, float *C, int M, int N, int K, cudaStream_t st) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(k_gemm_bi, cudaFuncAttributeMaxDynamicSharedMemorySize, GB_SMEM);
        init = true;
    }
    if (N % GB_BN || K % GB_BK) {
        fprintf(stderr, "pf: gemm_bi_nt: N %d / K %d not multiples of %d / %d\n", N, K, GB_BN, GB_BK);
        exit(1);
    }
    if (M <= 0) return;
    k_gemm_bi<<<dim3(N / GB_BN, (M + GB_BM - 1) / GB_BM), GB_THREADS, GB_SMEM, st>>>(X, W, C, M, N, K);
}
#endif
