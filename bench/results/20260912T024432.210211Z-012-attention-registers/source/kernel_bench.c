/* Synthetic-weight measurements of the actual production kernels.
 * Build through run.py. Every timed path includes pf.c; no copied SIMD code.
 * Reference calculations use independent double precision arithmetic.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define main pf_program_main
#ifndef PF_BENCH_SOURCE
#define PF_BENCH_SOURCE "../pf.c"
#endif
#include PF_BENCH_SOURCE
#undef main
#include <time.h>
#include <sys/resource.h>
#include <inttypes.h>

static int bench_parallel;
static uint32_t rng_state = 0x51a77e09u;
static float *initial_x, *proj_x, *proj_out;
static bf16 *proj_xh;
static int bench_experts, bench_m, bench_tokens, bench_segment;
static int bench_kind;
enum { BENCH_MOE, BENCH_QKV, BENCH_OUT, BENCH_PHASE_B, BENCH_LAYER };

static uint64_t ns_clock(clockid_t id) {
    struct timespec ts;
    if (clock_gettime(id, &ts)) die("clock_gettime failed");
    return (uint64_t)ts.tv_sec * 1000000000u + ts.tv_nsec;
}
static void *alloc_zero(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) die("benchmark allocation failed");
    return p;
}
static uint32_t random_u32(void) {
    uint32_t v = rng_state;
    v ^= v << 13; v ^= v >> 17; v ^= v << 5;
    return rng_state = v;
}
static float random_float(float scale) {
    return ((float)(random_u32() >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale;
}
static bf16 ref_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return (bf16)((u + 32767u + ((u >> 16) & 1u)) >> 16);
}
static double ref_from_bf16(bf16 h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return (double)f;
}
static double ref_activation_input(float f) {
#ifdef HAVE_BFDOT
    return ref_from_bf16(ref_to_bf16(f));
#else
    return (double)f;
#endif
}
static bf16 *make_weights(size_t n, float scale) {
    bf16 *p = alloc_zero(n, sizeof *p);
    for (size_t i = 0; i < n; i++) p[i] = ref_to_bf16(random_float(scale));
    return p;
}
static uint64_t hash_bytes(uint64_t h, const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static uint64_t hash_floats(uint64_t h, const float *p, size_t n) {
    return hash_bytes(h, p, n * sizeof *p);
}

static void setup_model(void) {
    const int max_tokens = 512;
    Layer *l = &M.L[0];
    buf_alloc(max_tokens);
    initial_x = alloc_zero((size_t)max_tokens * D, sizeof *initial_x);
    proj_x = alloc_zero(QD, sizeof *proj_x);
    proj_xh = alloc_zero(QD, sizeof *proj_xh);
    proj_out = alloc_zero(QKVD, sizeof *proj_out);
    /* All 128 experts are distinct and physically populated: 300 MiB. */
    l->w1 = make_weights((size_t)NE * D * 2 * FF, .045f);
    l->w2 = make_weights((size_t)NE * FF * D, .045f);
    l->b1 = make_weights((size_t)NE * 2 * FF, .05f);
    l->b2 = make_weights((size_t)NE * D, .05f);
    l->qkv_w = make_weights((size_t)QKVD * D, .045f);
    l->qkv_b = make_weights(QKVD, .05f);
    l->attn_out_w = make_weights((size_t)D * QD, .045f);
    l->attn_out_b = make_weights(D, .05f);
    l->gate_w = make_weights((size_t)NE * D, .045f);
    l->gate_b = make_weights(NE, .05f);
    l->attn_norm = alloc_zero(D, sizeof *l->attn_norm);
    l->mlp_norm = alloc_zero(D, sizeof *l->mlp_norm);
    l->sinks = alloc_zero(NH, sizeof *l->sinks);
    for (int i = 0; i < D; i++) {
        l->attn_norm[i] = ref_to_bf16(1.0f + random_float(.1f));
        l->mlp_norm[i] = ref_to_bf16(1.0f + random_float(.1f));
    }
    for (int i = 0; i < NH; i++) l->sinks[i] = random_float(1.0f);
    for (int i = 0; i < max_tokens * D; i++) {
        initial_x[i] = random_float(1.5f);
        R.tb[i] = random_float(1.5f);
        R.tbh[i] = ref_to_bf16(R.tb[i]);
    }
    for (int i = 0; i < max_tokens * QD; i++) R.q[i] = random_float(.35f);
    for (int i = 0; i < max_tokens * KVD; i++) {
        R.k[i] = random_float(.35f); R.v[i] = random_float(1.0f);
    }
    for (int i = 0; i < QD; i++) {
        proj_x[i] = random_float(1.5f); proj_xh[i] = ref_to_bf16(proj_x[i]);
    }
    for (int i = 0; i < max_tokens * TOPK; i++) R.gates[i] = .1f + (i % 4) * .1f;
    R.layer = 0;
    rope_init(max_tokens);
}

static void setup_case(int kind, int experts, int m, int tokens, int segment) {
    bench_kind = kind; bench_experts = experts; bench_m = m;
    bench_tokens = tokens; bench_segment = segment;
    R.T = tokens; R.layer = 0;
    if (kind == BENCH_MOE) {
        int n = 0, chunks = 0;
        for (int e = 0; e < experts; e++) {
            int start = n;
            for (int j = 0; j < m; j++) {
                R.wlist[n++] = (e * m + j) * TOPK;
                if (n-start == MOE_B) {
                    R.wchunk[chunks++] = (struct WChunk){e, start, MOE_B}; start = n;
                }
            }
            if (n > start) R.wchunk[chunks++] = (struct WChunk){e, start, n-start};
        }
        R.nchunk = chunks;
    }
    for (int t = 0; t < tokens; t++) {
        R.pos[t] = t % segment;
        R.sst[t] = t / segment * segment;
        int end = R.sst[t] + segment;
        R.sen[t] = end < tokens ? end : tokens;
    }
}
static void run_phase(size_t n, void (*fn)(void *, size_t)) {
    if (bench_parallel) pfor(n, fn, NULL);
    else for (size_t i = 0; i < n; i++) fn(NULL, i);
}
static void route_chunks(void) {
    for (int e = 0; e < NE; e++) R.bucket[e] = -1;
    for (int it = R.T * TOPK - 1; it >= 0; it--) {
        int e = R.eidx[it];
        if (e >= 0) { R.chain[it] = R.bucket[e]; R.bucket[e] = it; }
    }
    int w = 0, nc = 0;
    for (int e = 0; e < NE; e++) {
        int start = w;
        for (int it = R.bucket[e]; it >= 0; it = R.chain[it]) {
            R.wlist[w++] = it;
            if (w - start == MOE_B) {
                R.wchunk[nc++] = (struct WChunk){e, start, MOE_B}; start = w;
            }
        }
        if (w > start) R.wchunk[nc++] = (struct WChunk){e, start, w - start};
    }
    R.nchunk = nc;
}
static __attribute__((noinline)) void invoke_case(void) {
    Layer *l = &M.L[0];
    if (bench_kind == BENCH_MOE) run_phase(R.nchunk, ph_c);
    else if (bench_kind == BENCH_QKV) {
        for (int i = 0; i < QKVD; i++)
            proj_out[i] = DOT_X(proj_x, proj_xh, l->qkv_w + (size_t)i * D, D);
    } else if (bench_kind == BENCH_OUT) {
        for (int i = 0; i < D; i++)
            proj_out[i] = DOT_X(proj_x, proj_xh, l->attn_out_w + (size_t)i * QD, QD);
    } else {
        memcpy(R.x, initial_x, (size_t)R.T * D * sizeof(float));
        if (bench_kind == BENCH_LAYER) run_phase(R.T, ph_a);
        run_phase(R.T, ph_b);
        if (bench_kind == BENCH_LAYER) { route_chunks(); run_phase(R.nchunk, ph_c); }
    }
}

typedef struct { double max_abs, max_rel, sum_sq, sum_ref_sq; size_t count, failed; } Error;
static void compare_value(Error *e, float actual, double expected) {
    uint32_t bits;
    memcpy(&bits, &actual, sizeof bits);
    double d = fabs((double)actual - expected);
    double rel = d / fmax(fabs(expected), 1e-6);
    if (d > e->max_abs) e->max_abs = d;
    if (rel > e->max_rel) e->max_rel = rel;
    e->sum_sq += d*d; e->sum_ref_sq += expected*expected; e->count++;
    if ((bits & 0x7f800000u) == 0x7f800000u || d > 2e-5 + 2e-4 * fabs(expected)) e->failed++;
}
static void reference_expert(Error *err, int expert, int item) {
    Layer *l = &M.L[0];
    const bf16 *w1 = l->w1 + (size_t)expert * D * 2 * FF;
    const bf16 *w2 = l->w2 + (size_t)expert * FF * D;
    double pre[2 * FF], act[FF], out[D];
    for (int j = 0; j < 2 * FF; j++) pre[j] = ref_from_bf16(l->b1[expert * 2 * FF + j]);
    for (int i = 0; i < D; i++) {
        double a = ref_activation_input(R.tb[(size_t)(item / TOPK) * D + i]);
        for (int j = 0; j < 2 * FF; j++) pre[j] += a * ref_from_bf16(w1[(size_t)i * 2 * FF + j]);
    }
    for (int j = 0; j < FF; j++) {
        double g = fmin(pre[j], 7.0), v = fmin(fmax(pre[FF+j], -7.0), 7.0);
        act[j] = g / (1.0 + exp(-(double)SWIGLU_ALPHA * g)) * (v + 1.0);
#ifdef HAVE_BFDOT
        act[j] = ref_from_bf16(ref_to_bf16((float)act[j]));
#endif
    }
    for (int j = 0; j < D; j++) out[j] = ref_from_bf16(l->b2[expert * D + j]);
    for (int i = 0; i < FF; i++)
        for (int j = 0; j < D; j++) out[j] += act[i] * ref_from_bf16(w2[(size_t)i * D + j]);
    for (int j = 0; j < D; j++)
        compare_value(err, R.moe[(size_t)item * D+j], out[j] * R.gates[item]);
}
static void reference_attention(Error *err, int t) {
    Layer *l = &M.L[0];
    double ao[QD], x[D];
    int lo = t - BAND; if (lo < R.sst[t]) lo = R.sst[t];
    int hi = t + BAND; if (hi >= R.sen[t]) hi = R.sen[t] - 1;
    for (int h = 0; h < NH; h++) {
        int g = h / (NH / NKV);
        double scores[2 * BAND + 1], mx = l->sinks[h] * (double)(float)M_LN2;
        double sink = mx;
        for (int j = lo; j <= hi; j++) {
            double s = 0;
            for (int k = 0; k < HD; k++) s += (double)R.q[(size_t)t * QD+h*HD+k] * R.k[(size_t)j*KVD+g*HD+k];
            scores[j-lo] = s; mx = fmax(mx, s);
        }
        double den = exp(sink-mx);
        for (int j = lo; j <= hi; j++) { scores[j-lo] = exp(scores[j-lo]-mx); den += scores[j-lo]; }
        for (int k = 0; k < HD; k++) {
            double v = 0;
            for (int j = lo; j <= hi; j++) v += scores[j-lo]/den * R.v[(size_t)j*KVD+g*HD+k];
            ao[h*HD+k] = v;
            compare_value(err, R.ao[(size_t)t*QD+h*HD+k], v);
        }
    }
    double ss = 0, norm[D], logits[NE];
    for (int i = 0; i < D; i++) {
        x[i] = initial_x[(size_t)t*D+i] + ref_from_bf16(l->attn_out_b[i]);
        for (int j = 0; j < QD; j++)
            x[i] += ref_activation_input((float)ao[j]) * ref_from_bf16(l->attn_out_w[(size_t)i*QD+j]);
        compare_value(err, R.x[(size_t)t*D+i], x[i]);
        ss += x[i]*x[i];
    }
    for (int i = 0; i < D; i++) {
        double v = x[i] / sqrt(ss/D + RMS_EPS) * ref_from_bf16(l->mlp_norm[i]);
        compare_value(err, R.tb[(size_t)t*D+i], v);
        norm[i] = v;
    }
    for (int e = 0; e < NE; e++) {
        logits[e] = ref_from_bf16(l->gate_b[e]);
        for (int i = 0; i < D; i++)
            logits[e] += ref_activation_input((float)norm[i]) * ref_from_bf16(l->gate_w[(size_t)e*D+i]);
    }
    double top[TOPK], den = 0;
    for (int s = 0; s < TOPK; s++) {
        int best = 0;
        for (int e = 1; e < NE; e++) if (logits[e] > logits[best]) best = e;
        compare_value(err, (float)R.eidx[(size_t)t*TOPK+s], (double)best);
        top[s] = logits[best]; logits[best] = -1e30;
    }
    for (int s = TOPK-1; s >= 0; s--) { top[s] = exp(top[s]-top[0]); den += top[s]; }
    for (int s = 0; s < TOPK; s++) compare_value(err, R.gates[(size_t)t*TOPK+s], top[s]/den);
}
static Error correctness(void) {
    Error err = {0};
    if (bench_kind == BENCH_QKV || bench_kind == BENCH_OUT) {
        int rows = bench_kind == BENCH_QKV ? QKVD : D;
        int cols = bench_kind == BENCH_QKV ? D : QD;
        const bf16 *w = bench_kind == BENCH_QKV ? M.L[0].qkv_w : M.L[0].attn_out_w;
        for (int i = 0; i < rows; i++) {
            double s = 0;
            for (int j = 0; j < cols; j++) s += ref_activation_input(proj_x[j]) * ref_from_bf16(w[(size_t)i*cols+j]);
            compare_value(&err, proj_out[i], s);
        }
    } else if (bench_kind == BENCH_MOE) {
        /* All outputs for small blocks; deterministic first/middle/last chunks
         * for streaming cases, all rows and every item within those chunks. */
        for (int e = 0; e < bench_experts; e++)
            if (bench_experts <= 4 || e == 0 || e == bench_experts/2 || e == bench_experts-1)
                for (int j = 0; j < bench_m; j++) reference_expert(&err, e, (e*bench_m+j)*TOPK);
    } else {
        for (int t = 0; t < R.T; t++)
            if (t == 0 || t == R.T/2 || t == R.T-1) reference_attention(&err, t);
        if (bench_kind == BENCH_LAYER)
            for (int c = 0; c < R.nchunk; c++)
                if (c == 0 || c == R.nchunk/2 || c == R.nchunk-1)
                    reference_expert(&err, R.wchunk[c].e, R.wlist[R.wchunk[c].off]);
    }
    return err;
}
static uint64_t output_hash(void) {
    uint64_t h = UINT64_C(1469598103934665603);
    if (bench_kind == BENCH_QKV || bench_kind == BENCH_OUT)
        return hash_floats(h, proj_out, bench_kind == BENCH_QKV ? QKVD : D);
    if (bench_kind == BENCH_MOE) {
        for (int i = 0; i < bench_experts * bench_m; i++) h = hash_floats(h, R.moe+(size_t)i*TOPK*D, D);
        return h;
    }
    h = hash_floats(h, R.ao, (size_t)R.T*QD);
    h = hash_floats(h, R.x, (size_t)R.T*D);
    h = hash_floats(h, R.tb, (size_t)R.T*D);
    h = hash_floats(h, R.gates, (size_t)R.T*TOPK);
    h = hash_bytes(h, R.eidx, (size_t)R.T*TOPK*sizeof(int));
    if (bench_kind == BENCH_LAYER) h = hash_floats(h, R.moe, (size_t)R.T*TOPK*D);
    return h;
}
static double timeval_us(struct timeval v) { return (double)v.tv_sec*1e6 + v.tv_usec; }
static int run_case(const char *name, int kind, int experts, int m, int tokens,
                    int segment, int reps, double target_ms) {
    setup_case(kind, experts, m, tokens, segment);
    invoke_case();
    Error error = correctness();
    uint64_t expected_hash = output_hash();
    printf("{\"type\":\"correctness\",\"case\":\"%s\",\"checked_values\":%zu,\"failed_values\":%zu,"
           "\"max_absolute_error\":%.12g,\"max_relative_error\":%.12g,\"rmse\":%.12g,"
           "\"normalized_rmse\":%.12g,\"output_fnv1a64\":\"%016" PRIx64 "\","
           "\"absolute_tolerance\":2e-5,\"relative_tolerance\":2e-4}\n", name,
           error.count, error.failed, error.max_abs, error.max_rel,
           sqrt(error.sum_sq / fmax((double)error.count, 1.0)),
           sqrt(error.sum_sq / fmax(error.sum_ref_sq, 1e-30)), expected_hash);
    fflush(stdout);
    if (error.failed) return 1;
    uint64_t c0 = ns_clock(CLOCK_MONOTONIC);
    invoke_case(); invoke_case();
    uint64_t elapsed = ns_clock(CLOCK_MONOTONIC) - c0;
    int loops = (int)ceil(target_ms * 2e6 / (double)(elapsed ? elapsed : 1));
    if (loops < 1) loops = 1;
    if (loops > 10000) loops = 10000;
    size_t weight_bytes = (kind == BENCH_MOE || kind == BENCH_LAYER)
        ? (size_t)(kind == BENCH_MOE ? experts : NE)*(D*2*FF + FF*D)*sizeof(bf16)
        : (kind == BENCH_QKV ? (size_t)QKVD*D*2 : (size_t)D*QD*2 + (kind == BENCH_OUT ? 0 : NE*D*2));
    double flops = kind == BENCH_MOE ? (double)experts*m*2*(D*2*FF+FF*D)
        : kind == BENCH_QKV ? (double)QKVD*D*2
        : kind == BENCH_OUT ? (double)D*QD*2 : 0;
    printf("{\"type\":\"workload\",\"case\":\"%s\",\"synthetic_weights\":true,"
           "\"parallel\":%s,\"tokens\":%d,\"segment_tokens\":%d,\"experts\":%d,"
           "\"items_per_expert\":%d,\"weight_bytes\":%zu,\"matmul_flops_per_call\":%.0f,"
           "\"loops_per_rep\":%d,\"calibration_two_calls_ns\":%" PRIu64 "}\n",
           name, bench_parallel ? "true" : "false", tokens, segment, experts, m,
           weight_bytes, flops, loops, elapsed);
    for (int rep = 0; rep < reps; rep++) {
        struct rusage r0, r1;
        getrusage(RUSAGE_SELF, &r0);
        uint64_t cpu0 = ns_clock(CLOCK_PROCESS_CPUTIME_ID), wall0 = ns_clock(CLOCK_MONOTONIC);
        for (int i = 0; i < loops; i++) invoke_case();
        uint64_t wall = ns_clock(CLOCK_MONOTONIC) - wall0;
        uint64_t cpu = ns_clock(CLOCK_PROCESS_CPUTIME_ID) - cpu0;
        getrusage(RUSAGE_SELF, &r1);
        uint64_t h = output_hash();
        printf("{\"type\":\"sample\",\"case\":\"%s\",\"rep\":%d,\"loops\":%d,"
               "\"wall_ns\":%" PRIu64 ",\"process_cpu_ns\":%" PRIu64 ",\"ns_per_call\":%.6f,"
               "\"user_us\":%.0f,\"system_us\":%.0f,\"minor_faults\":%ld,\"major_faults\":%ld,"
               "\"voluntary_context_switches\":%ld,\"involuntary_context_switches\":%ld,"
               "\"max_rss_kib\":%ld,\"output_fnv1a64\":\"%016" PRIx64 "\",\"hash_stable\":%s}\n",
               name, rep, loops, wall, cpu, (double)wall/loops,
               timeval_us(r1.ru_utime)-timeval_us(r0.ru_utime),
               timeval_us(r1.ru_stime)-timeval_us(r0.ru_stime), r1.ru_minflt-r0.ru_minflt,
               r1.ru_majflt-r0.ru_majflt, r1.ru_nvcsw-r0.ru_nvcsw, r1.ru_nivcsw-r0.ru_nivcsw,
               r1.ru_maxrss, h, h == expected_hash ? "true" : "false");
        fflush(stdout);
        if (h != expected_hash) return 1;
    }
    return 0;
}
int main(int argc, char **argv) {
    int reps = 5, extended = 0;
    double target_ms = 25;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--parallel")) bench_parallel = 1;
        else if (!strcmp(argv[i], "--reps") && i+1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--target-ms") && i+1 < argc) target_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "--profile") && i+1 < argc) extended = !strcmp(argv[++i], "extended");
        else die("unknown benchmark option");
    }
    if (reps < 1 || reps > 1000 || target_ms <= 0 || target_ms > 10000) die("invalid benchmark settings");
    printf("{\"type\":\"metadata\",\"schema_version\":1,\"suite\":\"pf_synthetic_kernels_v1\","
           "\"seed_hex\":\"51a77e09\",\"model_weights\":false,\"D\":%d,\"FF\":%d,\"NE\":%d,\"MOE_B\":%d,"
           "\"reference\":\"independent_double\",\"clock\":\"CLOCK_MONOTONIC\",\"reps\":%d,"
           "\"target_ms\":%.3f,\"profile\":\"%s\"}\n", D, FF, NE, MOE_B, reps, target_ms,
           extended ? "extended" : "quick");
    setup_model();
    int failures = 0;
#define RUN(n,k,e,m,t,s) do { failures += run_case(n,k,e,m,t,s,reps,target_ms); } while (0)
    RUN("moe_e1_m1", BENCH_MOE, 1, 1, 1, 1);
    RUN("moe_e1_m4", BENCH_MOE, 1, 4, 4, 4);
    RUN("moe_e1_m16", BENCH_MOE, 1, 16, 16, 16);
    RUN("moe_e32_m1", BENCH_MOE, 32, 1, 32, 32);
    RUN("moe_e128_m1", BENCH_MOE, 128, 1, 128, 128);
    RUN("moe_e32_m4", BENCH_MOE, 32, 4, 128, 128);
    RUN("projection_qkv", BENCH_QKV, 0, 0, 1, 1);
    RUN("projection_out", BENCH_OUT, 0, 0, 1, 1);
    RUN("phase_b_t32", BENCH_PHASE_B, 0, 0, 32, 32);
    RUN("phase_b_t256", BENCH_PHASE_B, 0, 0, 256, 256);
    RUN("phase_b_packed_t256_s32", BENCH_PHASE_B, 0, 0, 256, 32);
    if (extended) {
        RUN("layer_t32", BENCH_LAYER, NE, 0, 32, 32);
        RUN("layer_packed_t256_s32", BENCH_LAYER, NE, 0, 256, 32);
    }
#undef RUN
    printf("{\"type\":\"completion\",\"correctness_failures\":%d}\n", failures);
    return failures ? 2 : 0;
}
