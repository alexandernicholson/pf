/* Deterministic correctness checks; no performance timing.
 *
 * Build from the repository root:
 *   cc -O3 -march=native bench/test_x86_moe.c -lm -pthread -o /tmp/test_x86_moe
 *   /tmp/test_x86_moe > /tmp/test_x86_moe.jsonl
 * Use -mavx2 -mfma for AVX2, -DPF_NO_AVX512_MOE to disable the AVX512 kernel,
 * -DPF_X86_MOE_KT=16 to check another reduction tile, or
 * -DPF_SOURCE='"/absolute/path/to/candidate.c"' -I. for a separate source copy.
 * An unavailable/disabled blocked kernel returns 77; a mismatch returns 1.
 */
#if defined(__FAST_MATH__)
#error "Correctness comparisons require strict floating-point compilation"
#endif
#ifndef PF_SOURCE
#define PF_SOURCE "../pf.c"
#endif
#define main pf_application_main
#include PF_SOURCE
#undef main
#include <inttypes.h>

#if defined(HAVE_X86_MOE)
static uint32_t test_rng = 42;
static float test_random(void) {
    test_rng ^= test_rng << 13;
    test_rng ^= test_rng >> 17;
    test_rng ^= test_rng << 5;
    return ((float)(test_rng % 20001) - 10000.0f) / 10000.0f;
}

static bf16 test_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return (bf16)(bits >> 16);
}

static uint64_t test_hash(uint64_t hash, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    for (int byte = 0; byte < 4; byte++) {
        hash ^= (bits >> (8 * byte)) & 255;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void *test_alloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) die("correctness check allocation failed");
    return p;
}

static int test_shape(int B, int n_in, int n_out, int pattern) {
    bf16 *w = test_alloc((size_t)n_in * n_out * sizeof *w);
    float *x[16], *y[16], *ref[16], *bias[16];
    for (int k = 0; k < n_in; k++) {
        for (int j = 0; j < n_out; j++) {
            float f = test_random();
            /* Mixed signs/scales and zeros exercise cancellation separately
             * from the dense ordinary-value case. All values remain finite. */
            if (pattern) f = (k % 7 == 0) ? 0.0f : ldexpf(f, (k % 17) - 8);
            w[(size_t)k * n_out + j] = test_bf16(f);
        }
    }
    for (int b = 0; b < B; b++) {
        x[b] = test_alloc((size_t)n_in * sizeof(float));
        y[b] = test_alloc((size_t)n_out * sizeof(float));
        ref[b] = test_alloc((size_t)n_out * sizeof(float));
        bias[b] = test_alloc((size_t)n_out * sizeof(float));
        for (int k = 0; k < n_in; k++) {
            float f = test_random();
            x[b][k] = pattern ? ldexpf(f, 8 - (k % 17)) : f;
        }
        for (int j = 0; j < n_out; j++)
            bias[b][j] = ref[b][j] = y[b][j] = test_random();
        for (int k = 0; k < n_in; k += 8)
            axpy8_bf16(x[b] + k, w + (size_t)k * n_out,
                       (size_t)n_out, ref[b], n_out);
    }
    moe_gemm_x86(w, (const float *const *)x, y, B, n_in, n_out);

    size_t mismatches = 0, scalar_mismatches = 0;
    double max_abs_error = 0;
    uint64_t output_hash = UINT64_C(14695981039346656037);
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < n_out; j++) {
            double delta = fabs((double)y[b][j] - ref[b][j]);
            if (delta > max_abs_error) max_abs_error = delta;
            if (memcmp(y[b] + j, ref[b] + j, sizeof(float))) mismatches++;
            output_hash = test_hash(output_hash, y[b][j]);
        }
        /* Independent scalar fused multiply-add oracle for tile boundaries,
         * including the first and last output lanes. */
        int columns[] = {0, 15, 16, 63, 64, n_out - 1};
        for (size_t q = 0; q < sizeof columns / sizeof *columns; q++) {
            int j = columns[q];
            float scalar = bias[b][j];
            for (int k = 0; k < n_in; k++)
                scalar = fmaf(x[b][k], b2f(w[(size_t)k * n_out + j]), scalar);
            if (memcmp(&scalar, y[b] + j, sizeof(float))) scalar_mismatches++;
        }
    }
    printf("{\"record\":\"correctness\",\"batch\":%d,\"n_in\":%d,"
           "\"n_out\":%d,\"pattern\":\"%s\",\"checked_outputs\":%zu,"
           "\"bitwise_mismatches\":%zu,\"scalar_checked_outputs\":%d,"
           "\"scalar_bitwise_mismatches\":%zu,\"max_abs_error\":%.17g,"
           "\"output_fnv1a64\":\"%016" PRIx64 "\"}\n",
           B, n_in, n_out, pattern ? "mixed_scale_sparse" : "dense",
           (size_t)B * n_out, mismatches, B * 6, scalar_mismatches,
           max_abs_error, output_hash);
    for (int b = 0; b < B; b++) {
        free(x[b]); free(y[b]); free(ref[b]); free(bias[b]);
    }
    free(w);
    return mismatches != 0 || scalar_mismatches != 0;
}
#endif

int main(void) {
#if defined(HAVE_X86_MOE)
    printf("{\"record\":\"metadata\",\"seed\":42,\"vector_lanes\":%d,"
           "\"output_tile\":%d,\"reduction_tile\":%d,\"moe_batch_limit\":%d,"
           "\"comparison\":\"bitwise\",\"tolerance\":0}\n",
           MOE_V, MOE_V * MOE_NV, PF_X86_MOE_KT, MOE_B);
    int batches[] = {1, 2, 3, 4, 7, 16};
    int cases = 0, failed = 0;
    for (int pattern = 0; pattern < 2; pattern++) {
        for (int shape = 0; shape < 2; shape++) {
            for (size_t b = 0; b < sizeof batches / sizeof *batches; b++) {
                failed += test_shape(batches[b], 640, shape ? 1280 : 640, pattern);
                cases++;
            }
        }
    }
    /* Three-token remainders after one, two, and three complete token
     * blocks, with partial reduction tiles for KT=16/32/64. These small
     * shapes cover both model output widths without repeating costly full
     * 640-row reductions already exercised above. */
    int tail_batches[] = {7, 11, 15};
    for (int shape = 0; shape < 2; shape++) {
        for (size_t b = 0; b < sizeof tail_batches / sizeof *tail_batches; b++) {
            failed += test_shape(tail_batches[b], 40, shape ? 1280 : 640, 1);
            cases++;
        }
    }
    printf("{\"record\":\"summary\",\"cases\":%d,\"failed_cases\":%d,"
           "\"status\":\"%s\"}\n", cases, failed, failed ? "fail" : "pass");
    return failed ? 1 : 0;
#else
    puts("{\"record\":\"summary\",\"status\":\"skip\","
         "\"reason\":\"x86 blocked kernel unavailable or disabled\"}");
    return 77;
#endif
}
