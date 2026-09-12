/* cc -O3 -march=native -DPF_X86_ATTN bench/test_x86_attention.c -lm -pthread
 * Define PF_SOURCE to a quoted alternate pf.c path for isolated candidates.
 * Emits JSONL correctness records; performs no performance timing. */
#if defined(__FAST_MATH__)
#error "Correctness comparisons require strict floating-point compilation"
#endif
#ifndef PF_SOURCE
#define PF_SOURCE "../pf.c"
#endif
#define main pf_application_main
#include PF_SOURCE
#undef main

#if defined(HAVE_X86_ATTN)
static uint32_t check_rng = 1729;
static float check_random(void) {
    check_rng ^= check_rng << 13;
    check_rng ^= check_rng >> 17;
    check_rng ^= check_rng << 5;
    return ((float)(check_rng % 20001) - 10000.0f) / 10000.0f;
}

static int check_window(int W, int group) {
    float sc[2 * BAND + 1], v[(2 * BAND + 1) * KVD];
    float expected[HD] = {0}, actual[HD];
    float den = expf(-3.5f);
    for (int j = 0; j < W; j++) {
        sc[j] = expf(check_random() * 25.0f - 25.0f);
        den += sc[j];
        for (int k = 0; k < KVD; k++)
            v[j * KVD + k] = ldexpf(check_random(), (k % 17) - 8);
    }
    /* The original ph_b traversal, including its per-position division. */
    for (int j = 0; j < W; j++) {
        float weight = sc[j] / den;
        const float *vj = v + j * KVD + group * HD;
        for (int i = 0; i < HD; i++) expected[i] += weight * vj[i];
    }
    attention_v_x86(sc, den, v + group * HD, W, actual);
    size_t mismatches = 0, scalar_mismatches = 0;
    double max_error = 0;
    for (int i = 0; i < HD; i++) {
        if (memcmp(actual + i, expected + i, sizeof(float))) mismatches++;
        double error = fabs((double)actual[i] - expected[i]);
        if (error > max_error) max_error = error;
        float scalar = 0;
        for (int j = 0; j < W; j++)
            scalar = fmaf(sc[j] / den, v[j * KVD + group * HD + i], scalar);
        if (memcmp(actual + i, &scalar, sizeof(float))) scalar_mismatches++;
    }
    printf("{\"record\":\"correctness\",\"window\":%d,\"kv_group\":%d,"
           "\"checked_outputs\":%d,\"bitwise_mismatches\":%zu,"
           "\"scalar_bitwise_mismatches\":%zu,\"max_abs_error\":%.17g}\n",
           W, group, HD, mismatches, scalar_mismatches, max_error);
    return mismatches != 0 || scalar_mismatches != 0;
}
#endif

int main(void) {
#if defined(HAVE_X86_ATTN)
    puts("{\"record\":\"metadata\",\"seed\":1729,\"vector_lanes\":8,"
         "\"comparison\":\"bitwise\",\"tolerance\":0}");
    int windows[] = {0, 1, 2, 3, 15, 16, 17, 128, 129, 256, 257};
    int failed = 0, cases = 0;
    for (int group = 0; group < NKV; group++) {
        for (size_t w = 0; w < sizeof windows / sizeof *windows; w++) {
            failed += check_window(windows[w], group);
            cases++;
        }
    }
    printf("{\"record\":\"summary\",\"cases\":%d,\"failed_cases\":%d,"
           "\"status\":\"%s\"}\n", cases, failed, failed ? "fail" : "pass");
    return failed ? 1 : 0;
#else
    puts("{\"record\":\"summary\",\"status\":\"skip\","
         "\"reason\":\"x86 attention kernel unavailable or disabled\"}");
    return 77;
#endif
}
