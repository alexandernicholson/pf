/* gemm_bench.cu — candidates/serve2/gemm_bi.cu vs cuBLAS (the call gpu.cu makes) on pf's dense shapes.
 *
 * Per shape (N x K = 1152x640 qkv, 640x896 out, 128x640 router):
 *   timing  : M in {131072, 65536, 16384, 4096, 1024, 160, 64}; 3 alternating rounds of
 *             (cuBLAS x REPS, gemm_bi x REPS) after warmup, event-timed; median ms per call
 *   accuracy: at M = 131072 both vs an fp64 host reference on 256 sampled rows (max |err| and
 *             max |err| / sum|a*b|), and gemm_bi vs cuBLAS max |diff|
 *   invariance: rows [off, off+m) recomputed as their own M (m in {64,160,1000,4096}, off in
 *             {0,1,37,1000}) must equal the M = 131072 result bitwise; also reported for cuBLAS
 *             (informational: cuBLAS chooses its kernel by M)
 *   config identity: none needed (one tile config); fill patterns NaN-poison C before each call
 * One JSON object per line on stdout; exit 1 if gemm_bi fails accuracy or invariance.
 * build: nvcc -O3 -arch=sm_90a -std=c++17 -I../../../candidates/serve2 gemm_bench.cu
 *        ../../../candidates/serve2/gemm_bi.cu -lcublas -o gemm_bench
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include "gemm_bi.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(2); } } while (0)

static cublasHandle_t cb;
static void gemm_cublas(const uint16_t *X, const uint16_t *W, float *C, int M, int N, int K) {
    const float one = 1.0f, zero = 0.0f;
    CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, C,
                    CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
static void gemm_mine(const uint16_t *X, const uint16_t *W, float *C, int M, int N, int K) {
    gemm_bi_nt(X, W, C, M, N, K, 0);
}

static uint64_t rs = 99;
static double urand() {
    rs += 0x9E3779B97F4A7C15ull;
    uint64_t z = rs;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return ((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
}
static float nrand() { return (float)(sqrt(-2 * log(urand() + 1e-300)) * cos(6.283185307179586 * urand())); }
static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

typedef void (*GemmFn)(const uint16_t *, const uint16_t *, float *, int, int, int);
static float time_ms(GemmFn f, const uint16_t *X, const uint16_t *W, float *C, int M, int N, int K, int reps) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) f(X, W, C, M, N, K);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / reps;
}
static float med(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main() {
    const int MMAX = 131072;
    const int shapes[3][2] = {{1152, 640}, {640, 896}, {128, 640}};
    const char *names[3] = {"qkv", "out", "router"};
    const int Ms[] = {131072, 65536, 16384, 4096, 1024, 160, 64};
    CB(cublasCreate(&cb));
    int fails = 0;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("{\"device\":\"%s\",\"sms\":%d}\n", p.name, p.multiProcessorCount);
    for (int s = 0; s < 3; s++) {
        const int N = shapes[s][0], K = shapes[s][1];
        std::vector<uint16_t> hX((size_t)MMAX * K), hW((size_t)N * K);
        for (auto &v : hX) v = f2b(nrand());
        for (auto &v : hW) v = f2b(0.05f * nrand());
        uint16_t *X, *W; float *C1, *C2, *Cs;
        CK(cudaMalloc(&X, hX.size() * 2)); CK(cudaMalloc(&W, hW.size() * 2));
        CK(cudaMalloc(&C1, (size_t)MMAX * N * 4)); CK(cudaMalloc(&C2, (size_t)MMAX * N * 4));
        CK(cudaMalloc(&Cs, (size_t)4096 * N * 4));
        CK(cudaMemcpy(X, hX.data(), hX.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(W, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));

        /* accuracy at full M */
        CK(cudaMemset(C1, 0xFF, (size_t)MMAX * N * 4)); CK(cudaMemset(C2, 0xFF, (size_t)MMAX * N * 4));
        gemm_cublas(X, W, C1, MMAX, N, K); gemm_mine(X, W, C2, MMAX, N, K);
        CK(cudaDeviceSynchronize());
        std::vector<float> h1((size_t)MMAX * N), h2((size_t)MMAX * N);
        CK(cudaMemcpy(h1.data(), C1, h1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), C2, h2.size() * 4, cudaMemcpyDeviceToHost));
        double e1 = 0, e2 = 0, r1 = 0, r2 = 0, d12 = 0; long nan2 = 0, eq12 = 0;
        for (size_t i = 0; i < h2.size(); i++) {
            nan2 += !(h2[i] == h2[i]);
            d12 = std::max(d12, (double)fabsf(h1[i] - h2[i]));
            eq12 += h1[i] == h2[i];
        }
        for (int k = 0; k < 256; k++) {
            int m = (int)(urand() * MMAX);
            for (int n = 0; n < N; n++) {
                double ref = 0, mag = 0;
                for (int j = 0; j < K; j++) {
                    double t = (double)b2f(hX[(size_t)m * K + j]) * b2f(hW[(size_t)n * K + j]);
                    ref += t; mag += fabs(t);
                }
                double a1 = fabs(h1[(size_t)m * N + n] - ref), a2 = fabs(h2[(size_t)m * N + n] - ref);
                e1 = std::max(e1, a1); e2 = std::max(e2, a2);
                r1 = std::max(r1, a1 / mag); r2 = std::max(r2, a2 / mag);
            }
        }
        /* fp32 accumulation over K terms: |err| <= K * 2^-24 * sum|a*b| (loose); NaN = fail */
        bool acc_ok = nan2 == 0 && r2 <= K * 5.96e-8;
        printf("{\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%d,\"bi_max_abs_err\":%.3e,\"bi_max_rel_err\":%.3e,"
               "\"cublas_max_abs_err\":%.3e,\"cublas_max_rel_err\":%.3e,\"bi_vs_cublas_max_abs\":%.3e,"
               "\"bi_eq_cublas_frac\":%.6f,\"bi_nan\":%ld,\"accuracy_pass\":%s}\n", names[s], N, K, MMAX, e2, r2, e1, r1,
               d12, (double)eq12 / h2.size(), nan2, acc_ok ? "true" : "false");
        fails += !acc_ok;

        /* invariance: sub-batches vs the full-M result */
        const int subs[] = {64, 160, 1000, 4096}, offs[] = {0, 1, 37, 1000};
        long mism_bi = 0, mism_cb = 0, nvals = 0;
        std::vector<float> hs((size_t)4096 * N);
        for (int m : subs)
            for (int o : offs) {
                for (int which = 0; which < 2; which++) {
                    CK(cudaMemset(Cs, 0xFF, (size_t)4096 * N * 4));
                    (which ? gemm_mine : gemm_cublas)(X + (size_t)o * K, W, Cs, m, N, K);
                    CK(cudaMemcpy(hs.data(), Cs, (size_t)m * N * 4, cudaMemcpyDeviceToHost));
                    const float *full = (which ? h2.data() : h1.data()) + (size_t)o * N;
                    long mm = 0;
                    for (size_t i = 0; i < (size_t)m * N; i++) mm += memcmp(&hs[i], &full[i], 4) != 0;
                    (which ? mism_bi : mism_cb) += mm;
                }
                nvals += (long)m * N;
            }
        printf("{\"shape\":\"%s\",\"invariance_values\":%ld,\"bi_mismatch\":%ld,\"cublas_mismatch\":%ld,"
               "\"bi_invariant\":%s}\n", names[s], nvals, mism_bi, mism_cb, mism_bi == 0 ? "true" : "false");
        fails += mism_bi != 0;

        /* timing */
        for (int M : Ms) {
            int reps = M >= 16384 ? 20 : 200;
            for (int w = 0; w < 3; w++) { gemm_cublas(X, W, C1, M, N, K); gemm_mine(X, W, C2, M, N, K); }
            std::vector<float> tc, tb;
            for (int r = 0; r < 3; r++) {
                tc.push_back(time_ms(gemm_cublas, X, W, C1, M, N, K, reps));
                tb.push_back(time_ms(gemm_mine, X, W, C2, M, N, K, reps));
            }
            float mc = med(tc), mb = med(tb);
            double fl = 2.0 * M * N * K;
            printf("{\"shape\":\"%s\",\"M\":%d,\"cublas_ms\":%.5f,\"bi_ms\":%.5f,\"bi_over_cublas\":%.3f,"
                   "\"cublas_tflops\":%.1f,\"bi_tflops\":%.1f}\n", names[s], M, mc, mb, mb / mc, fl / mc / 1e9,
                   fl / mb / 1e9);
        }
        fflush(stdout);
        CK(cudaFree(X)); CK(cudaFree(W)); CK(cudaFree(C1)); CK(cudaFree(C2)); CK(cudaFree(Cs));
    }
    CK(cudaGetLastError());
    printf("{\"gemm_bench_pass\":%s}\n", fails ? "false" : "true");
    return fails ? 1 : 0;
}
