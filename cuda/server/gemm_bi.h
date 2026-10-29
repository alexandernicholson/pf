/* gemm_bi.h — batch-invariant dense GEMM for pf's projections (PF_BATCH_INVARIANT=1).
 *
 * C[M][N] fp32 = X[M][K] . W[N][K]^T, X and W bf16 (pf's dense weights are
 * [out][in]), fp32 accumulation. Tensor-core mma.sync m16n8k16 with one fixed
 * tile (128 x 128 x 32, 4-stage cp.async pipeline, no split-K), so each output
 * element is the same chain of k16 MMAs in ascending K order whatever M is,
 * whichever CTA computes the row, and whatever rows are next to it. A row's
 * result is therefore bit-identical for every batch composition (cuBLAS picks
 * its kernel and split by M, which is what made the server differ from the
 * batch path).
 * Requires N % 128 == 0 and K % 32 == 0 (pf: 1152x640, 640x896, 128x640).
 */
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
void gemm_bi_nt(const uint16_t *X, const uint16_t *W, float *C, int M, int N, int K, cudaStream_t st);
#ifdef __cplusplus
}
#endif
