/* pf.c — OpenAI Privacy Filter (openai/privacy-filter) CPU inference in one file.
 *
 * Reimplements the reference Python stack (github.com/openai/privacy-filter):
 *   o200k_base BPE tokenizer -> bidirectional banded-attention MoE transformer
 *   (bf16 weights mmap'd in place, fp32 math) -> log_softmax -> constrained
 *   BIOES Viterbi -> span masking.
 *
 * Build: make        Run: ./pf "Hi Jordan, call me at +1 (415) 555-0124."
 * Weights: ./model/{config.json,model.safetensors,o200k_base.tiktoken}
 *          (original-format checkpoint from HF openai/privacy-filter, original/)
 */
#define _DEFAULT_SOURCE   /* glibc/musl: expose usleep, strncasecmp under -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <pthread.h>

#if defined(__aarch64__)
#include <arm_neon.h>
/* SME2 code is built when the target already has SME2, or on clang via a
 * per-function target region (GCC's target pragma leaks SVE into the rest of
 * the file); it is used only if the CPU and OS support it (sme2_usable). */
#if defined(__ARM_FEATURE_BF16) && !defined(PF_NO_BFDOT) && !defined(PF_NO_SME) && defined(__has_include)
#if __has_include(<arm_sme.h>) && (defined(__ARM_FEATURE_SME2) || defined(__clang__))
#define HAVE_SME 1
#endif
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define HAVE_SDOT 1                /* --int8 expert GEMMs on NEON SDOT */
#endif
#endif
#if defined(HAVE_SME) && defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(HAVE_SME) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP2_SME2
#define HWCAP2_SME2 (1UL << 37)
#endif
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

#include "unicode_tables.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16) && !defined(PF_NO_BFDOT)
#define DOT_X(xf, xh, w, n)            dot_bfx(xh, w, n)
#define AXPY8_X(xf, xh, w, stride, y, n) axpy8_bfx(xh, w, stride, y, n)
#define CVT_H(x, y, n)                 cvt_f32_bf16(x, y, n)
#else
#define DOT_X(xf, xh, w, n)            dot_bf16(xf, w, n)
#define AXPY8_X(xf, xh, w, stride, y, n) axpy8_bf16(xf, w, stride, y, n)
#define CVT_H(x, y, n)                 ((void)(y))
#endif

/* ------------------------------------------------------------------ config */
#define D      640   /* d_model */
#define NL     8     /* layers */
#define NH     14    /* query heads */
#define NKV    2     /* kv heads */
#define HD     64    /* head dim */
#define QD     (NH * HD)          /* 896 */
#define KVD    (NKV * HD)         /* 128 */
#define QKVD   (QD + 2 * KVD)     /* 1152 */
#define NE     128   /* experts */
#define TOPK   4
#define FF     640   /* intermediate size per expert */
#define NC     33    /* label classes */
#define NCAT   8     /* span categories */
#define BAND   128   /* bidirectional left/right context */
#define VOCAB  200064
#define ROPE_THETA   150000.0
#define YARN_FACTOR  32.0
#define YARN_ORIG    4096.0
#define YARN_ALPHA   1.0
#define YARN_BETA    32.0
#define SWIGLU_ALPHA 1.702f
#define SWIGLU_LIMIT 7.0f
#define RMS_EPS      1e-5f
#define QK_SCALE     0.35355339059327379f /* 64^-0.25 */
#define NEG_INF      (-1e9f)
#ifndef MOE_B
#define MOE_B  16    /* tokens per expert block */
#endif
#define DB     16    /* tokens per SME block (dense projections, attention) */
#define KW     (DB + 2 * BAND)   /* widest union key window of one SME block */
#define Q8_FIRST 2   /* --int8 quantizes the experts of layers >= this (see q8_pack) */

static float g_skip_beta = 0.0f;   /* --skip-beta: dynamic expert skipping */
static int   g_ff = FF;            /* expert intermediate width; < FF after --trim */
static double (*g_calib)[NE][FF];  /* --calib: per-neuron sum of gate * |swiglu out| */
static int    g_q8;                /* int8 expert weights and activations (layers >= Q8_FIRST) */
static pthread_mutex_t g_calib_mu = PTHREAD_MUTEX_INITIALIZER;
static const char *CAT[NCAT] = {"account_number","private_address","private_date",
    "private_email","private_person","private_phone","private_url","secret"};

typedef uint16_t bf16;

static void die(const char *msg) { fprintf(stderr, "pf: %s\n", msg); exit(1); }
static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

/* ------------------------------------------------------------ SIMD kernels */
static inline float b2f(bf16 h) {
    union { uint32_t u; float f; } v; v.u = (uint32_t)h << 16; return v.f;
}

#if defined(__aarch64__)
/* dot(x_f32[n], w_bf16[n]) */
static __attribute__((unused)) float dot_bf16(const float *x, const bf16 *w, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t h0 = vld1q_u16(w + i), h1 = vld1q_u16(w + i + 8);
        a0 = vfmaq_f32(a0, vld1q_f32(x + i),
                       vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16)));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4),
                       vreinterpretq_f32_u32(vshll_high_n_u16(h0, 16)));
        a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8),
                       vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16)));
        a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12),
                       vreinterpretq_f32_u32(vshll_high_n_u16(h1, 16)));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += x[i] * b2f(w[i]);
    return s;
}
/* y[n] += a * w_bf16[n] */
static __attribute__((unused)) void axpy_bf16(float a, const bf16 *w, float *y, int n) {
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(w + i);
        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(vshll_high_n_u16(h, 16));
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     va, w0));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), va, w1));
    }
    for (; i < n; i++) y[i] += a * b2f(w[i]);
}
static float dot_f32(const float *x, const float *y, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 = vfmaq_f32(a0, vld1q_f32(x + i), vld1q_f32(y + i));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vld1q_f32(y + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; i++) s += x[i] * y[i];
    return s;
}
/* y[j] += sum_r a[r] * w[r*stride + j], 8 rows fused; n % 16 == 0 */
static __attribute__((unused)) void axpy8_bf16(const float *a, const bf16 *w, size_t stride, float *y, int n) {
    float32x4_t va[8];
    for (int r = 0; r < 8; r++) va[r] = vdupq_n_f32(a[r]);
    for (int j = 0; j < n; j += 16) {
        float32x4_t y0 = vld1q_f32(y + j),      y1 = vld1q_f32(y + j + 4);
        float32x4_t y2 = vld1q_f32(y + j + 8),  y3 = vld1q_f32(y + j + 12);
        float32x4_t z0 = vdupq_n_f32(0), z1 = vdupq_n_f32(0);
        float32x4_t z2 = vdupq_n_f32(0), z3 = vdupq_n_f32(0);
        for (int r = 0; r < 8; r += 2) {
            const bf16 *w0 = w + r * stride + j, *w1 = w0 + stride;
            uint16x8_t ha = vld1q_u16(w0), hb = vld1q_u16(w0 + 8);
            uint16x8_t hc = vld1q_u16(w1), hd = vld1q_u16(w1 + 8);
            y0 = vfmaq_f32(y0, va[r], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(ha), 16)));
            y1 = vfmaq_f32(y1, va[r], vreinterpretq_f32_u32(vshll_high_n_u16(ha, 16)));
            y2 = vfmaq_f32(y2, va[r], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(hb), 16)));
            y3 = vfmaq_f32(y3, va[r], vreinterpretq_f32_u32(vshll_high_n_u16(hb, 16)));
            z0 = vfmaq_f32(z0, va[r+1], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(hc), 16)));
            z1 = vfmaq_f32(z1, va[r+1], vreinterpretq_f32_u32(vshll_high_n_u16(hc, 16)));
            z2 = vfmaq_f32(z2, va[r+1], vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(hd), 16)));
            z3 = vfmaq_f32(z3, va[r+1], vreinterpretq_f32_u32(vshll_high_n_u16(hd, 16)));
        }
        vst1q_f32(y + j,      vaddq_f32(y0, z0));
        vst1q_f32(y + j + 4,  vaddq_f32(y1, z1));
        vst1q_f32(y + j + 8,  vaddq_f32(y2, z2));
        vst1q_f32(y + j + 12, vaddq_f32(y3, z3));
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static inline __m256 bf8(const bf16 *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
static float dot_bf16(const float *x, const bf16 *w, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), bf8(w + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), bf8(w + i + 8), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += x[i] * b2f(w[i]);
    return s;
}
static float dot_f32(const float *x, const float *y, int n) {
    __m256 a0 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), a0);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += x[i] * y[i];
    return s;
}
static void axpy8_bf16(const float *a, const bf16 *w, size_t stride, float *y, int n) {
    for (int j = 0; j < n; j += 16) {
        __m256 y0 = _mm256_loadu_ps(y + j), y1 = _mm256_loadu_ps(y + j + 8);
        for (int r = 0; r < 8; r++) {
            __m256 va = _mm256_set1_ps(a[r]);
            const bf16 *wr = w + r * stride + j;
            y0 = _mm256_fmadd_ps(va, bf8(wr), y0);
            y1 = _mm256_fmadd_ps(va, bf8(wr + 8), y1);
        }
        _mm256_storeu_ps(y + j, y0); _mm256_storeu_ps(y + j + 8, y1);
    }
}
static __attribute__((unused)) void axpy_bf16(float a, const bf16 *w, float *y, int n) {
    int i = 0;
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, bf8(w + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * b2f(w[i]);
}
#else
static __attribute__((unused)) float dot_bf16(const float *x, const bf16 *w, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += x[i] * b2f(w[i]); return s;
}
static __attribute__((unused)) void axpy_bf16(float a, const bf16 *w, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] += a * b2f(w[i]);
}
static float dot_f32(const float *x, const float *y, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s;
}
static __attribute__((unused)) void axpy8_bf16(const float *a, const bf16 *w, size_t stride, float *y, int n) {
    for (int r = 0; r < 8; r++) axpy_bf16(a[r], w + r * stride, y, n);
}
#endif

/* copy bf16 -> f32 */
#if defined(__aarch64__)
static void cvt_bf16(const bf16 *w, float *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t h = vld1q_u16(w + i);
        vst1q_f32(y + i,     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16)));
        vst1q_f32(y + i + 4, vreinterpretq_f32_u32(vshll_high_n_u16(h, 16)));
    }
    for (; i < n; i++) y[i] = b2f(w[i]);
}
/* exp for 4 lanes, Cephes-style: ~1 ulp fp32, |x| clamped to +-87 */
static inline float32x4_t exp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(87.0f));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    float32x4_t r2 = vmulq_f32(r, r);
    p = vaddq_f32(vfmaq_f32(r, p, r2), vdupq_n_f32(1.0f));
    int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e));
}
#elif defined(__AVX2__)
static void cvt_bf16(const bf16 *w, float *y, int n);   /* defined after kernels */
#define CVT_BF16_AVX2 1
#else
static void cvt_bf16(const bf16 *w, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = b2f(w[i]);
}
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16) && !defined(PF_NO_BFDOT)
/* BFDOT kernels: weights stay mmap'd bf16, activations rounded to bf16 (RNE),
 * products exact, accumulation fp32 — matches the reference's bf16 semantics. */
#define HAVE_BFDOT 1
static inline bfloat16x8_t ldbf(const bf16 *p) { return vld1q_bf16((const bfloat16_t *)p); }

static void cvt_f32_bf16(const float *x, bf16 *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        bfloat16x8_t v = vcombine_bf16(vcvt_bf16_f32(vld1q_f32(x + i)),
                                       vcvt_bf16_f32(vld1q_f32(x + i + 4)));
        vst1q_bf16((bfloat16_t *)(y + i), v);
    }
    for (; i < n; i++) { float f = x[i]; uint32_t u; memcpy(&u, &f, 4);
        uint32_t r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16; y[i] = (bf16)r; }
}

/* dot of two bf16 vectors, fp32 accumulate */
static float dot_bfx(const bf16 *x, const bf16 *w, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = vbfdotq_f32(a0, ldbf(x + i),      ldbf(w + i));
        a1 = vbfdotq_f32(a1, ldbf(x + i + 8),  ldbf(w + i + 8));
        a2 = vbfdotq_f32(a2, ldbf(x + i + 16), ldbf(w + i + 16));
        a3 = vbfdotq_f32(a3, ldbf(x + i + 24), ldbf(w + i + 24));
    }
    for (; i + 8 <= n; i += 8) a0 = vbfdotq_f32(a0, ldbf(x + i), ldbf(w + i));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += b2f(x[i]) * b2f(w[i]);
    return s;
}

/* Blocked expert GEMM: ys[i][j] += sum_k xs[i][k] * w[k*n_out + j] for i < B.
 * Per 8-row k-tile the weight rows are pair-interleaved once into an L1-resident
 * buffer, then swept by up to MOE_B tokens (BFDOT lane form), so mmap'd expert
 * weights stream from L2/DRAM once per 16-token block instead of once per token. */
static void moe_gemm(const bf16 *w, const bf16 *const *xs, float *const *ys,
                     int B, int n_in, int n_out) {
    bf16 zbuf[8 * 2 * FF];
    for (int kt = 0; kt < n_in; kt += 8) {
        const bf16 *wk = w + (size_t)kt * n_out;
        for (int j = 0; j < n_out; j += 8) {
            uint16x8_t r0 = vld1q_u16(wk + j),             r1 = vld1q_u16(wk + n_out + j);
            uint16x8_t r2 = vld1q_u16(wk + 2 * n_out + j), r3 = vld1q_u16(wk + 3 * n_out + j);
            uint16x8_t r4 = vld1q_u16(wk + 4 * n_out + j), r5 = vld1q_u16(wk + 5 * n_out + j);
            uint16x8_t r6 = vld1q_u16(wk + 6 * n_out + j), r7 = vld1q_u16(wk + 7 * n_out + j);
            bf16 *z = zbuf + (size_t)j * 8;
            vst1q_u16(z,      vzip1q_u16(r0, r1)); vst1q_u16(z + 8,  vzip2q_u16(r0, r1));
            vst1q_u16(z + 16, vzip1q_u16(r2, r3)); vst1q_u16(z + 24, vzip2q_u16(r2, r3));
            vst1q_u16(z + 32, vzip1q_u16(r4, r5)); vst1q_u16(z + 40, vzip2q_u16(r4, r5));
            vst1q_u16(z + 48, vzip1q_u16(r6, r7)); vst1q_u16(z + 56, vzip2q_u16(r6, r7));
        }
        for (int i = 0; i < B; i++) {
            bfloat16x8_t xp = ldbf(xs[i] + kt);
            float *y = ys[i];
            const bf16 *z = zbuf;
            for (int j = 0; j < n_out; j += 8, z += 64) {
                float32x4_t a0 = vld1q_f32(y + j), a1 = vld1q_f32(y + j + 4);
                a0 = vbfdotq_laneq_f32(a0, ldbf(z),      xp, 0);
                a1 = vbfdotq_laneq_f32(a1, ldbf(z + 8),  xp, 0);
                a0 = vbfdotq_laneq_f32(a0, ldbf(z + 16), xp, 1);
                a1 = vbfdotq_laneq_f32(a1, ldbf(z + 24), xp, 1);
                a0 = vbfdotq_laneq_f32(a0, ldbf(z + 32), xp, 2);
                a1 = vbfdotq_laneq_f32(a1, ldbf(z + 40), xp, 2);
                a0 = vbfdotq_laneq_f32(a0, ldbf(z + 48), xp, 3);
                a1 = vbfdotq_laneq_f32(a1, ldbf(z + 56), xp, 3);
                vst1q_f32(y + j, a0); vst1q_f32(y + j + 4, a1);
            }
        }
    }
}

#if defined(HAVE_SME)
/* clang target region: streaming-mode functions only, so no SVE reaches
 * normal code (M4 has no non-streaming SVE) */
#if !defined(__ARM_FEATURE_SME2)
#pragma clang attribute push(__attribute__((target("sme2"))), apply_to = function)
#endif
#include <arm_sme.h>

/* SME2 expert GEMM: y[MP][N] = x * W, fp32 rows of stride N.
 * x packed [K/2][MP][2], W packed [K/2][N][2] (k-pair interleaved bf16) so one
 * BFMOPA consumes 16 tokens x 16 outputs x 2 k. One token tile x four column
 * tiles per pass; MP % 16 == 0, K even, N % 16 == 0. Runs in streaming mode,
 * where NEON is unavailable, so callers do bias/activation outside. */
__arm_locally_streaming __arm_new("za")
static void sme_gemm(const bf16 *xp, int MP, const bf16 *wp, float *y, int K, int N) {
    const svbool_t pt = svptrue_b16(), p32 = svptrue_b32();
    const svcount_t pc = svptrue_c16();
    for (int t0 = 0; t0 < MP; t0 += 16) {
        int n0 = 0;
        for (; n0 + 64 <= N; n0 += 64) {
            svzero_za();
            const bf16 *a = xp + t0 * 2, *b = wp + n0 * 2;
            for (int p = 0; p < K / 2; p++, a += MP * 2, b += N * 2) {
                svbfloat16_t va = svld1_bf16(pt, (const bfloat16_t *)a);
                svbfloat16x4_t vb = svld1_bf16_x4(pc, (const bfloat16_t *)b);
                svmopa_za32_bf16_m(0, pt, pt, va, svget4(vb, 0));
                svmopa_za32_bf16_m(1, pt, pt, va, svget4(vb, 1));
                svmopa_za32_bf16_m(2, pt, pt, va, svget4(vb, 2));
                svmopa_za32_bf16_m(3, pt, pt, va, svget4(vb, 3));
            }
            for (int r = 0; r < 16; r++) {
                float *o = y + (size_t)(t0 + r) * N + n0;
                svst1_f32(p32, o,      svread_hor_za32_f32_m(svundef_f32(), p32, 0, r));
                svst1_f32(p32, o + 16, svread_hor_za32_f32_m(svundef_f32(), p32, 1, r));
                svst1_f32(p32, o + 32, svread_hor_za32_f32_m(svundef_f32(), p32, 2, r));
                svst1_f32(p32, o + 48, svread_hor_za32_f32_m(svundef_f32(), p32, 3, r));
            }
        }
        for (; n0 < N; n0 += 16) {      /* leftover column tiles (narrow --trim widths) */
            svzero_za();
            const bf16 *a = xp + t0 * 2, *b = wp + n0 * 2;
            for (int p = 0; p < K / 2; p++, a += MP * 2, b += N * 2)
                svmopa_za32_bf16_m(0, pt, pt, svld1_bf16(pt, (const bfloat16_t *)a),
                                   svld1_bf16(pt, (const bfloat16_t *)b));
            for (int r = 0; r < 16; r++)
                svst1_f32(p32, y + (size_t)(t0 + r) * N + n0,
                          svread_hor_za32_f32_m(svundef_f32(), p32, 0, r));
        }
    }
}

/* int8 expert GEMM (--int8): y[MP][N] = x * W, int32 rows of stride N.
 * x [K/4][MP][4], W [K/4][N][4]: one SMOPA consumes 16 tokens x 16 outputs x
 * 4 k, exact int32 accumulation. Same tiling as sme_gemm. */
__arm_locally_streaming __arm_new("za")
static void sme_gemm_s8(const int8_t *xp, int MP, const int8_t *wp, int32_t *y, int K, int N) {
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svcount_t pc = svptrue_c8();
    for (int t0 = 0; t0 < MP; t0 += 16) {
        int n0 = 0;
        for (; n0 + 64 <= N; n0 += 64) {
            svzero_za();
            const int8_t *a = xp + t0 * 4, *b = wp + n0 * 4;
            for (int p = 0; p < K / 4; p++, a += MP * 4, b += N * 4) {
                svint8_t va = svld1_s8(p8, a);
                svint8x4_t vb = svld1_s8_x4(pc, b);
                svmopa_za32_s8_m(0, p8, p8, va, svget4(vb, 0));
                svmopa_za32_s8_m(1, p8, p8, va, svget4(vb, 1));
                svmopa_za32_s8_m(2, p8, p8, va, svget4(vb, 2));
                svmopa_za32_s8_m(3, p8, p8, va, svget4(vb, 3));
            }
            for (int r = 0; r < 16; r++) {
                int32_t *o = y + (size_t)(t0 + r) * N + n0;
                svst1_s32(p32, o,      svread_hor_za32_s32_m(svundef_s32(), p32, 0, r));
                svst1_s32(p32, o + 16, svread_hor_za32_s32_m(svundef_s32(), p32, 1, r));
                svst1_s32(p32, o + 32, svread_hor_za32_s32_m(svundef_s32(), p32, 2, r));
                svst1_s32(p32, o + 48, svread_hor_za32_s32_m(svundef_s32(), p32, 3, r));
            }
        }
        for (; n0 < N; n0 += 16) {
            svzero_za();
            const int8_t *a = xp + t0 * 4, *b = wp + n0 * 4;
            for (int p = 0; p < K / 4; p++, a += MP * 4, b += N * 4)
                svmopa_za32_s8_m(0, p8, p8, svld1_s8(p8, a), svld1_s8(p8, b));
            for (int r = 0; r < 16; r++)
                svst1_s32(p32, y + (size_t)(t0 + r) * N + n0,
                          svread_hor_za32_s32_m(svundef_s32(), p32, 0, r));
        }
    }
}

/* fp32 FMOPA GEMM for attention (which stays fp32, like the reference):
 * y[MP][N] (row stride ldy) = sum_k a[k] (x) b[k]; row k of a (stride lda)
 * holds the MP output rows' k-th values, row k of b (stride ldb) the N
 * columns'. MP % 16 == 0, N % 16 == 0. Four column tiles per pass when
 * N >= 64, four row tiles when N == 16 (the K Q^T shape). */
__arm_locally_streaming __arm_new("za")
static void sme_gemm_f32(const float *a, int lda, int MP, const float *b, int ldb,
                         float *y, int ldy, int K, int N) {
    const svbool_t p32 = svptrue_b32();
    const svcount_t pc = svptrue_c32();
    int t0 = 0;
    if (N == 16)
        for (; t0 + 64 <= MP; t0 += 64) {
            svzero_za();
            for (int k = 0; k < K; k++) {
                svfloat32x4_t va = svld1_f32_x4(pc, a + (size_t)k * lda + t0);
                svfloat32_t vb = svld1_f32(p32, b + (size_t)k * ldb);
                svmopa_za32_f32_m(0, p32, p32, svget4(va, 0), vb);
                svmopa_za32_f32_m(1, p32, p32, svget4(va, 1), vb);
                svmopa_za32_f32_m(2, p32, p32, svget4(va, 2), vb);
                svmopa_za32_f32_m(3, p32, p32, svget4(va, 3), vb);
            }
            for (int r = 0; r < 16; r++) {
                float *o = y + (size_t)(t0 + r) * ldy;
                svst1_f32(p32, o,            svread_hor_za32_f32_m(svundef_f32(), p32, 0, r));
                svst1_f32(p32, o + 16 * ldy, svread_hor_za32_f32_m(svundef_f32(), p32, 1, r));
                svst1_f32(p32, o + 32 * ldy, svread_hor_za32_f32_m(svundef_f32(), p32, 2, r));
                svst1_f32(p32, o + 48 * ldy, svread_hor_za32_f32_m(svundef_f32(), p32, 3, r));
            }
        }
    for (; t0 < MP; t0 += 16) {
        int n0 = 0;
        for (; n0 + 64 <= N; n0 += 64) {
            svzero_za();
            for (int k = 0; k < K; k++) {
                svfloat32_t va = svld1_f32(p32, a + (size_t)k * lda + t0);
                svfloat32x4_t vb = svld1_f32_x4(pc, b + (size_t)k * ldb + n0);
                svmopa_za32_f32_m(0, p32, p32, va, svget4(vb, 0));
                svmopa_za32_f32_m(1, p32, p32, va, svget4(vb, 1));
                svmopa_za32_f32_m(2, p32, p32, va, svget4(vb, 2));
                svmopa_za32_f32_m(3, p32, p32, va, svget4(vb, 3));
            }
            for (int r = 0; r < 16; r++) {
                float *o = y + (size_t)(t0 + r) * ldy + n0;
                svst1_f32(p32, o,      svread_hor_za32_f32_m(svundef_f32(), p32, 0, r));
                svst1_f32(p32, o + 16, svread_hor_za32_f32_m(svundef_f32(), p32, 1, r));
                svst1_f32(p32, o + 32, svread_hor_za32_f32_m(svundef_f32(), p32, 2, r));
                svst1_f32(p32, o + 48, svread_hor_za32_f32_m(svundef_f32(), p32, 3, r));
            }
        }
        for (; n0 < N; n0 += 16) {
            svzero_za();
            for (int k = 0; k < K; k++)
                svmopa_za32_f32_m(0, p32, p32, svld1_f32(p32, a + (size_t)k * lda + t0),
                                  svld1_f32(p32, b + (size_t)k * ldb + n0));
            for (int r = 0; r < 16; r++)
                svst1_f32(p32, y + (size_t)(t0 + r) * ldy + n0,
                          svread_hor_za32_f32_m(svundef_f32(), p32, 0, r));
        }
    }
}

__arm_locally_streaming static int sme_svl_words(void) { return (int)svcntsw(); }
#if !defined(__ARM_FEATURE_SME2)
#pragma clang attribute pop
#endif

/* run-time check: CPU + OS support SME2, with the 512-bit streaming vectors
 * (16 fp32 lanes) that sme_gemm's tile layout assumes */
static int sme2_usable(void) {
    int hw = 0;
#if defined(__APPLE__)
    size_t n = sizeof hw;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &hw, &n, NULL, 0)) hw = 0;
#elif defined(__linux__)
    hw = (getauxval(AT_HWCAP2) & HWCAP2_SME2) != 0;
#endif
    return hw && sme_svl_words() == 16;
}

/* [K][N] -> [K/2][N][2]: rows 2p and 2p+1 interleaved per column */
static void pack_kpairs(const bf16 *s, bf16 *d, int K, int N) {
    for (int p = 0; p < K / 2; p++)
        for (int j = 0; j < N; j++) {
            d[((size_t)p * N + j) * 2]     = s[(size_t)(2 * p) * N + j];
            d[((size_t)p * N + j) * 2 + 1] = s[(size_t)(2 * p + 1) * N + j];
        }
}

/* [N][K] (output-major, as stored) -> [K/2][N][2]; memcpy: tensors may be 2-byte aligned */
static void pack_omajor(const bf16 *w, int N, int K, bf16 *d) {
    uint32_t *d32 = (uint32_t *)d;
    for (int o = 0; o < N; o++)
        for (int p = 0; p < K / 2; p++)
            memcpy(&d32[(size_t)p * N + o], w + (size_t)o * K + 2 * p, 4);
}

/* m bf16 rows of n -> [n/2][MP][2], rows m..MP zero-padded */
static void pack_rows(const bf16 *const *rows, int m, int MP, int n, bf16 *d) {
    uint32_t *d32 = (uint32_t *)d;
    for (int i = 0; i < MP; i++) {
        if (i < m) {
            const uint32_t *r = (const uint32_t *)rows[i];
            for (int p = 0; p < n / 2; p++) d32[(size_t)p * MP + i] = r[p];
        } else {
            for (int p = 0; p < n / 2; p++) d32[(size_t)p * MP + i] = 0;
        }
    }
}
#endif
#endif

/* int8 expert GEMM on the vector unit, same operands as sme_gemm_s8 (rows
 * m..MP of x must be zero or are ignored; only rows < m are guaranteed). On
 * NEON a 16-byte W load is 4 columns x 4 k: the SDOT-by-lane operand, with
 * each lane of an x load one token's 4 k. 8 tokens x 8 columns per pass. */
#if defined(HAVE_SDOT)
static void gemm_s8_vec(const int8_t *xp, int MP, int m, const int8_t *wp, int32_t *y, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 8)
        for (int t0 = 0; t0 < m; t0 += 8) {
            int32x4_t c[8][2];
            for (int i = 0; i < 8; i++) c[i][0] = c[i][1] = vdupq_n_s32(0);
            const int8_t *a = xp + t0 * 4, *b = wp + n0 * 4;
            for (int p = 0; p < K / 4; p++, a += MP * 4, b += N * 4) {
                const int8x16_t x0 = vld1q_s8(a), x1 = vld1q_s8(a + 16);
                const int8x16_t w0 = vld1q_s8(b), w1 = vld1q_s8(b + 16);
#define SD(i, x, l) c[i][0] = vdotq_laneq_s32(c[i][0], w0, x, l); \
                    c[i][1] = vdotq_laneq_s32(c[i][1], w1, x, l);
                SD(0, x0, 0) SD(1, x0, 1) SD(2, x0, 2) SD(3, x0, 3)
                SD(4, x1, 0) SD(5, x1, 1) SD(6, x1, 2) SD(7, x1, 3)
#undef SD
            }
            for (int i = 0; i < 8; i++) {
                vst1q_s32(y + (size_t)(t0 + i) * N + n0,     c[i][0]);
                vst1q_s32(y + (size_t)(t0 + i) * N + n0 + 4, c[i][1]);
            }
        }
}
#else
static void gemm_s8_vec(const int8_t *xp, int MP, int m, const int8_t *wp, int32_t *y, int K, int N) {
    for (int t = 0; t < m; t++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int p = 0; p < K / 4; p++)
                for (int j = 0; j < 4; j++)
                    s += xp[((size_t)p * MP + t) * 4 + j] * wp[((size_t)p * N + n) * 4 + j];
            y[(size_t)t * N + n] = s;
        }
}
#endif

#if defined(CVT_BF16_AVX2)
static void cvt_bf16(const bf16 *w, float *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, bf8(w + i));
    for (; i < n; i++) y[i] = b2f(w[i]);
}
#endif

/* --------------------------------------------------------------- parallel */
static int pf_serial = -1;

/* threads that run model code get 8 MiB stacks: the kernels keep ~150 KB of
 * per-block buffers on the stack, and musl's default thread stack is 128 KB */
static void spawn(void *(*fn)(void *), void *arg, int detached) {
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, (size_t)8 << 20);
    if (detached) pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&th, &at, fn, arg)) die("pthread_create failed");
    pthread_attr_destroy(&at);
}

#if !defined(__APPLE__) || defined(PF_POOL)
/* portable pthread worker pool (Linux/Graviton; libdispatch is Apple-only) */
#include <stdatomic.h>
static struct {
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    void (*fn)(void *, size_t); void *ctx;
    size_t n; _Atomic size_t next;
    int epoch, running, nthreads, started;
} PP = { .mu = PTHREAD_MUTEX_INITIALIZER,
         .go = PTHREAD_COND_INITIALIZER, .done = PTHREAD_COND_INITIALIZER };

static void *pp_worker(void *arg) {
    (void)arg;
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&PP.mu);
        while (PP.epoch == seen) pthread_cond_wait(&PP.go, &PP.mu);
        seen = PP.epoch;
        pthread_mutex_unlock(&PP.mu);
        for (size_t i; (i = atomic_fetch_add(&PP.next, 1)) < PP.n; )
            PP.fn(PP.ctx, i);
        pthread_mutex_lock(&PP.mu);
        if (--PP.running == 0) pthread_cond_signal(&PP.done);
        pthread_mutex_unlock(&PP.mu);
    }
    return NULL;
}

static void pool_pfor(size_t n, void (*fn)(void *, size_t), void *ctx) {
    if (!PP.started) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        PP.nthreads = nc < 1 ? 1 : (nc > 64 ? 64 : (int)nc);
        for (int i = 0; i < PP.nthreads; i++)
            spawn(pp_worker, NULL, 1);
        PP.started = 1;
    }
    pthread_mutex_lock(&PP.mu);
    PP.fn = fn; PP.ctx = ctx; PP.n = n;
    atomic_store(&PP.next, 0);
    PP.running = PP.nthreads;
    PP.epoch++;
    pthread_cond_broadcast(&PP.go);
    while (PP.running) pthread_cond_wait(&PP.done, &PP.mu);
    pthread_mutex_unlock(&PP.mu);
}
#endif

static void pfor(size_t n, void (*fn)(void *, size_t), void *ctx) {
    if (pf_serial < 0) pf_serial = getenv("PF_SERIAL") != NULL;
    if (!pf_serial) {
#if defined(__APPLE__) && !defined(PF_POOL)
        dispatch_apply_f(n, DISPATCH_APPLY_AUTO, ctx, fn);
#else
        pool_pfor(n, fn, ctx);
#endif
        return;
    }
    for (size_t i = 0; i < n; i++) fn(ctx, i);
}

/* ------------------------------------------------------------- safetensors */
static uint8_t *st_base;   /* start of tensor data area */
static char    *st_hdr;    /* JSON header (NUL-terminated copy) */
static size_t  st_size;

#define TOUCH_CHUNK (8u << 20)
static void touch_pages(void *ctx, size_t i) {
    volatile const uint8_t *p = (const uint8_t *)ctx + i * TOUCH_CHUNK;
    size_t n = st_size - i * TOUCH_CHUNK;
    if (n > TOUCH_CHUNK) n = TOUCH_CHUNK;
    volatile uint8_t s = 0;
    for (size_t o = 0; o < n; o += 16384) s ^= p[o];
    (void)s;
}

static void st_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open model.safetensors");
    struct stat sb; fstat(fd, &sb);
    uint64_t hlen;
    if (read(fd, &hlen, 8) != 8) die("bad safetensors");
    void *m = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) die("mmap failed");
    close(fd);
    st_hdr = malloc(hlen + 1);
    memcpy(st_hdr, (uint8_t *)m + 8, hlen);
    st_hdr[hlen] = 0;
    st_base = (uint8_t *)m + 8 + hlen;
    st_size = (size_t)sb.st_size;
    madvise(m, st_size, MADV_WILLNEED);
    pfor((st_size + TOUCH_CHUNK - 1) / TOUCH_CHUNK, touch_pages, m);
    mlock(m, st_size);   /* best-effort pin; matters for long-running use */
}

/* find tensor by name; verify dtype/element count; return pointer */
static void *st_get(const char *name, const char *dtype, size_t numel) {
    char key[128];
    snprintf(key, sizeof key, "\"%s\":", name);
    char *p = strstr(st_hdr, key);
    if (!p) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p += strlen(key);
    char *dt = strstr(p, "\"dtype\":\"");
    if (!dt || strncmp(dt + 9, dtype, strlen(dtype)) != 0)
        { fprintf(stderr, "dtype mismatch for %s\n", name); exit(1); }
    char *sh = strstr(p, "\"shape\":[");
    size_t n = 1;
    for (char *q = sh + 9; *q && *q != ']'; ) {
        n *= strtoull(q, &q, 10);
        while (*q == ',' || *q == ' ') q++;
    }
    if (n != numel) { fprintf(stderr, "shape mismatch for %s\n", name); exit(1); }
    char *off = strstr(p, "\"data_offsets\":[");
    uint64_t o = strtoull(off + 16, NULL, 10);
    return st_base + o;
}

/* ------------------------------------------------------------------ model */
typedef struct {
    bf16 *attn_norm, *qkv_w, *qkv_b, *attn_out_w, *attn_out_b;
    float *sinks;                      /* [NH] F32, log2 space */
    bf16 *mlp_norm, *gate_w, *gate_b;
    bf16 *w1, *b1;                     /* [NE][D][2W], [NE][2W]  (W = g_ff) */
    bf16 *w2, *b2;                     /* [NE][W][D],  [NE][D]  */
    bf16 *qkv_p, *out_p, *gate_p;      /* SME packings of qkv_w, attn_out_w, gate_w */
} Layer;

static struct {
    bf16 *emb;                         /* [VOCAB][D] */
    Layer L[NL];
    bf16 *fnorm, *unemb;               /* [D], [NC][D] */
} M;

static void model_load(const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    st_open(path);
    M.emb   = st_get("embedding.weight", "BF16", (size_t)VOCAB * D);
    M.fnorm = st_get("norm.scale", "BF16", D);
    M.unemb = st_get("unembedding.weight", "BF16", (size_t)NC * D);
    for (int l = 0; l < NL; l++) {
        char nm[64]; Layer *y = &M.L[l];
#define GET(field, suffix, dt, n) \
        (snprintf(nm, sizeof nm, "block.%d.%s", l, suffix), \
         y->field = st_get(nm, dt, n))
        GET(attn_norm,  "attn.norm.scale", "BF16", D);
        GET(qkv_w,      "attn.qkv.weight", "BF16", (size_t)QKVD * D);
        GET(qkv_b,      "attn.qkv.bias",   "BF16", QKVD);
        GET(sinks,      "attn.sinks",      "F32",  NH);
        GET(attn_out_w, "attn.out.weight", "BF16", (size_t)D * QD);
        GET(attn_out_b, "attn.out.bias",   "BF16", D);
        GET(mlp_norm,   "mlp.norm.scale",  "BF16", D);
        GET(gate_w,     "mlp.gate.weight", "BF16", (size_t)NE * D);
        GET(gate_b,     "mlp.gate.bias",   "BF16", NE);
        GET(w1,         "mlp.swiglu.weight","BF16",(size_t)NE * D * 2 * FF);
        GET(b1,         "mlp.swiglu.bias", "BF16", (size_t)NE * 2 * FF);
        GET(w2,         "mlp.out.weight",  "BF16", (size_t)NE * FF * D);
        GET(b2,         "mlp.out.bias",    "BF16", (size_t)NE * D);
#undef GET
    }
}

/* --trim: keep g_ff of the FF intermediate neurons of every expert, re-packing
 * the expert weights densely so the MoE GEMMs shrink by g_ff/FF. Mask file:
 * "PFTRIM1\0", u32 width, then NL*NE*FF keep bits (LSB first), built by trim.py. */
static void trim_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open --trim mask");
    static uint8_t bits[NL * NE * FF / 8];
    char magic[8]; uint32_t w = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PFTRIM1", 8) ||
        fread(&w, 4, 1, f) != 1 || fread(bits, 1, sizeof bits, f) != sizeof bits || fgetc(f) != EOF)
        die("bad --trim mask file");
    fclose(f);
    if (w == 0 || w > FF || w % 8) die("--trim width must be a multiple of 8 in 8..640");
    for (int l = 0; l < NL; l++) {
        Layer *y = &M.L[l];
        bf16 *w1 = malloc((size_t)NE * D * 2 * w * 2), *b1 = malloc((size_t)NE * 2 * w * 2);
        bf16 *w2 = malloc((size_t)NE * w * D * 2);
        if (!w1 || !b1 || !w2) die("out of memory");
        for (int e = 0; e < NE; e++) {
            int idx[FF], n = 0;
            for (int j = 0; j < FF; j++) {
                size_t b = ((size_t)l * NE + e) * FF + j;
                if (bits[b >> 3] >> (b & 7) & 1) idx[n++] = j;
            }
            if (n != (int)w) die("--trim mask: kept-neuron count differs from width");
            const bf16 *s1 = y->w1 + (size_t)e * D * 2 * FF, *sb = y->b1 + (size_t)e * 2 * FF;
            const bf16 *s2 = y->w2 + (size_t)e * FF * D;
            bf16 *d1 = w1 + (size_t)e * D * 2 * w, *db = b1 + (size_t)e * 2 * w;
            for (int k = 0; k < D; k++)
                for (int i = 0; i < n; i++) {   /* gate half, then linear half */
                    d1[(size_t)k * 2 * w + i]     = s1[(size_t)k * 2 * FF + idx[i]];
                    d1[(size_t)k * 2 * w + w + i] = s1[(size_t)k * 2 * FF + FF + idx[i]];
                }
            for (int i = 0; i < n; i++) {
                db[i] = sb[idx[i]]; db[w + i] = sb[FF + idx[i]];
                memcpy(w2 + ((size_t)e * w + i) * D, s2 + (size_t)idx[i] * D, D * sizeof(bf16));
            }
        }
        y->w1 = w1; y->b1 = b1; y->w2 = w2;
    }
    g_ff = (int)w;
}

static void unpin(const void *p, size_t n) {
    uintptr_t pg = (uintptr_t)getpagesize();
    uintptr_t a = ((uintptr_t)p + pg - 1) & ~(pg - 1), b = ((uintptr_t)p + n) & ~(pg - 1);
    if (b > a) { munlock((void *)a, b - a); madvise((void *)a, b - a, MADV_DONTNEED); }
}

#if defined(HAVE_SME)
/* SME (used when sme2_usable(); --exact keeps the reference-matching BFDOT path):
 * re-pack expert weights k-pair interleaved (sme_gemm layout) once at load,
 * after --trim; the mmap'd originals are unpinned so RSS doesn't double. */
static int g_sme;
static bf16 *g_pk[NL][2];
static void pack_expert(void *ctx, size_t i) {
    (void)ctx;
    const int l = (int)(i / NE), W = g_ff;
    const size_t e = i % NE;
    const Layer *y = &M.L[l];
    pack_kpairs(y->w1 + e * D * 2 * W, g_pk[l][0] + e * D * 2 * W, D, 2 * W);
    pack_kpairs(y->w2 + e * W * D,     g_pk[l][1] + e * W * D,     W, D);
}
static void sme_pack(void) {
    if (!sme2_usable()) return;
    const size_t W = (size_t)g_ff;
    const int nl = g_q8 ? Q8_FIRST : NL;   /* --int8 replaced the later experts */
    for (int l = 0; l < nl; l++)
        if (!(g_pk[l][0] = malloc(NE * D * 2 * W * 2)) || !(g_pk[l][1] = malloc(NE * W * D * 2)))
            die("out of memory");
    pfor((size_t)nl * NE, pack_expert, NULL);
    for (int l = 0; l < NL; l++) {     /* dense projections, stored [out][in] */
        Layer *y = &M.L[l];
        if (!(y->qkv_p = malloc((size_t)QKVD * D * 2)) || !(y->out_p = malloc((size_t)D * QD * 2)) ||
            !(y->gate_p = malloc((size_t)NE * D * 2)))
            die("out of memory");
        pack_omajor(y->qkv_w, QKVD, D, y->qkv_p);
        pack_omajor(y->attn_out_w, D, QD, y->out_p);
        pack_omajor(y->gate_w, NE, D, y->gate_p);
    }
    for (int l = 0; l < nl; l++) {
        Layer *y = &M.L[l];
        if (W != FF) { free(y->w1); free(y->w2); }   /* trim_load's copies */
        else { unpin(y->w1, NE * D * 2 * W * 2); unpin(y->w2, NE * W * D * 2); }
        y->w1 = g_pk[l][0]; y->w2 = g_pk[l][1];
    }
    g_sme = 1;
}
#else
#define sme_pack() ((void)0)
#define g_sme 0
#define sme2_usable() 0
#endif

/* --int8 (layers >= Q8_FIRST): expert GEMMs in int8 with exact int32
 * accumulation. Both GEMM inputs are first rotated by an orthonormal
 * Walsh-Hadamard transform H along k (x W = (x H)(H W), H = H^T; the weight
 * side is folded in at load), which spreads outlier channels so max/rms of
 * a row drops to the Gaussian ~3.3 (from 4-20 for x, 11-17 for the SwiGLU
 * output). Then symmetric int8: weights per output column, activations per
 * row, scale max|v| / 127. Weights are stored [K/4][N][4] for sme_gemm_s8 /
 * gemm_s8_vec; both kernels give identical results. Layers 0-1 stay bf16:
 * quantizing either one alone moves the output 3-10x more than any later
 * layer does. */
static int8_t *g_w8[NL][2];
static float  *g_s8[NL][2];
/* in-place orthonormal Walsh-Hadamard transform on blocks of the largest
 * power of two (<= 128) dividing n: n rows of stride s, cols columns each */
static int had_blk(int n) { int b = n & -n; return b > 128 ? 128 : b; }
static void fwht(float *x, int n, size_t s, int cols) {
    const int B = had_blk(n);
    const float sc = 1.0f / sqrtf((float)B);
    for (int o = 0; o < n; o += B)
        for (int len = 1; len < B; len <<= 1)
            for (int i = o; i < o + B; i += 2 * len)
                for (int j = i; j < i + len; j++) {
                    float *a = x + (size_t)j * s, *b = x + (size_t)(j + len) * s;
                    for (int c = 0; c < cols; c++) { float u = a[c], v = b[c]; a[c] = u + v; b[c] = u - v; }
                }
    for (int j = 0; j < n; j++)
        for (int c = 0; c < cols; c++) x[(size_t)j * s + c] *= sc;
}
/* fwht of one contiguous row (n % 8 == 0): butterflies of span 1 and 2 inside
 * each 4-lane vector, wider spans between vectors */
static void fwht_row(float *x, int n) {
#if defined(__aarch64__)
    const int B = had_blk(n);
    for (int i = 0; i < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i), r = vrev64q_f32(v);
        v = vtrn1q_f32(vaddq_f32(v, r), vsubq_f32(v, r));          /* span 1 */
        float32x4_t w = vextq_f32(v, v, 2);
        vst1q_f32(x + i, vcombine_f32(vget_low_f32(vaddq_f32(v, w)),
                                      vget_low_f32(vsubq_f32(v, w)))); /* span 2 */
    }
    for (int len = 4; len < B; len <<= 1)
        for (int o = 0; o < n; o += 2 * len)
            for (int j = o; j < o + len; j += 4) {
                float32x4_t a = vld1q_f32(x + j), b = vld1q_f32(x + j + len);
                vst1q_f32(x + j, vaddq_f32(a, b));
                vst1q_f32(x + j + len, vsubq_f32(a, b));
            }
    const float32x4_t sc = vdupq_n_f32(1.0f / sqrtf((float)B));
    for (int i = 0; i < n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), sc));
#else
    fwht(x, n, 1, 1);
#endif
}
static void q8_mat(const bf16 *w, int K, int N, int8_t *d, float *s) {   /* w [K][N] */
    float mx[2 * FF] = {0};
    float *f = malloc((size_t)K * N * 4);
    if (!f) die("out of memory");
    for (size_t i = 0; i < (size_t)K * N; i++) f[i] = b2f(w[i]);
    fwht(f, K, (size_t)N, N);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) mx[n] = fmaxf(mx[n], fabsf(f[(size_t)k * N + n]));
    for (int n = 0; n < N; n++) s[n] = mx[n] > 0 ? mx[n] / 127 : 1;
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            d[((size_t)(k / 4) * N + n) * 4 + k % 4] = (int8_t)lrintf(f[(size_t)k * N + n] / s[n]);
    free(f);
}
static void q8_expert(void *ctx, size_t i) {
    (void)ctx;
    const int l = Q8_FIRST + (int)(i / NE), W = g_ff;
    const size_t e = i % NE;
    const Layer *y = &M.L[l];
    q8_mat(y->w1 + e * D * 2 * W, D, 2 * W, g_w8[l][0] + e * D * 2 * W, g_s8[l][0] + e * 2 * W);
    q8_mat(y->w2 + e * W * D,     W, D,     g_w8[l][1] + e * W * D,     g_s8[l][1] + e * D);
}
static void q8_pack(void) {
    const size_t W = (size_t)g_ff;
    for (int l = Q8_FIRST; l < NL; l++)
        if (!(g_w8[l][0] = malloc(NE * D * 2 * W)) || !(g_w8[l][1] = malloc(NE * W * D)) ||
            !(g_s8[l][0] = malloc(NE * 2 * W * 4)) || !(g_s8[l][1] = malloc(NE * D * 4)))
            die("out of memory");
    pfor((size_t)(NL - Q8_FIRST) * NE, q8_expert, NULL);
    for (int l = Q8_FIRST; l < NL; l++) {
        Layer *y = &M.L[l];
        if (W != FF) { free(y->w1); free(y->w2); }   /* trim_load's copies */
        else { unpin(y->w1, NE * D * 2 * W * 2); unpin(y->w2, NE * W * D * 2); }
        y->w1 = y->w2 = NULL;
    }
}
static void gemm_s8(const int8_t *xp, int MP, int m, const int8_t *wp, int32_t *y, int K, int N) {
#if defined(HAVE_SME)
    if (g_sme) { sme_gemm_s8(xp, MP, wp, y, K, N); return; }
#endif
    gemm_s8_vec(xp, MP, m, wp, y, K, N);
}
/* symmetric per-row int8 (n % 8 == 0): q = round(x * 127 / max|x|), *s = max|x| / 127 */
static void q8_row(const float *x, int n, int8_t *q, float *s) {
#if defined(__aarch64__)
    float32x4_t vm = vdupq_n_f32(0);
    for (int i = 0; i < n; i += 4) vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(x + i)));
    const float mx = vmaxvq_f32(vm);
    const float32x4_t inv = vdupq_n_f32(mx > 0 ? 127.0f / mx : 0.0f);
    for (int i = 0; i < n; i += 8) {       /* round to nearest even, like rintf */
        int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i), inv));
        int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4), inv));
        vst1_s8(q + i, vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))));
    }
#else
    float mx = 0;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(x[i]));
    const float inv = mx > 0 ? 127.0f / mx : 0.0f;
    for (int i = 0; i < n; i++) q[i] = (int8_t)(int32_t)rintf(x[i] * inv);
#endif
    *s = mx / 127.0f;
}

/* --calib: per-neuron importance E[gate * |h_j|] * ||W2 row j||, normalized per
 * token so stats from several corpora can be summed with equal weight (trim.py).
 * A tiny ||W2|| term breaks ties for neurons the corpus never reached. Needs
 * the bf16 experts: main rejects --calib with --trim or --int8. */
static void calib_write(const char *path, size_t ntok) {
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write --calib stats");
    for (int l = 0; l < NL; l++)
        for (int e = 0; e < NE; e++)
            for (int j = 0; j < FF; j++) {
                const bf16 *w2 = M.L[l].w2 + (size_t)e * FF * D;
                double n2 = 0;
                for (int i = 0; i < D; i++) {   /* row j of [FF][D], or of its SME [FF/2][D][2] packing */
                    size_t k = g_sme ? ((size_t)(j / 2) * D + i) * 2 + (j & 1) : (size_t)j * D + i;
                    float v = b2f(w2[k]);
                    n2 += (double)v * v;
                }
                float s = (float)((g_calib[l][e][j] / (double)(ntok ? ntok : 1) + 1e-12) * sqrt(n2));
                fwrite(&s, sizeof s, 1, f);
            }
    if (fclose(f)) die("cannot write --calib stats");
}

/* --------------------------------------------------------- unicode classes */
#define FL_L  1
#define FL_N  2
#define FL_U  4   /* Lu|Lt|Lm|Lo|M  */
#define FL_D  8   /* Ll|Lm|Lo|M     */
#define FL_WS 16

static int uc_flags(uint32_t cp) {
    int lo = 0, hi = UC_NRUNS;
    while (lo + 1 < hi) {           /* last run with start <= cp */
        int mid = (lo + hi) / 2;
        if (uc_start[mid] <= cp) lo = mid; else hi = mid;
    }
    return uc_flag[lo];
}

/* decode one UTF-8 codepoint at s[0..n); returns bytes consumed (1 on error) */
static int utf8_dec(const uint8_t *s, size_t n, uint32_t *cp) {
    uint8_t c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
    if (len == 1 || (size_t)len > n) { *cp = 0xFFFD; return 1; }
    static const uint32_t mask[5] = {0, 0x7F, 0x1F, 0x0F, 0x07};
    uint32_t v = c & mask[len];
    for (int i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (s[i] & 0x3F);
    }
    *cp = v;
    return len;
}

/* -------------------------------------------------------------- tokenizer */
#define HASH_BITS 20
#define HASH_SLOTS (1u << HASH_BITS)
static struct { uint32_t off; int32_t rank; uint16_t len; } tk_slot[HASH_SLOTS];
static uint8_t *tk_arena;
static int tk_max_len;

static uint64_t fnv1a(const uint8_t *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 0x100000001b3ULL; }
    return h;
}

static int32_t tk_rank(const uint8_t *s, size_t n) {
    if (n == 0 || (int)n > tk_max_len) return -1;
    uint64_t h = fnv1a(s, n);
    for (uint32_t i = (uint32_t)(h & (HASH_SLOTS - 1)); ; i = (i + 1) & (HASH_SLOTS - 1)) {
        if (tk_slot[i].rank < 0) return -1;
        if (tk_slot[i].len == n && memcmp(tk_arena + tk_slot[i].off, s, n) == 0)
            return tk_slot[i].rank;
    }
}

static void tk_insert(const uint8_t *s, size_t n, int32_t rank, uint32_t arena_off) {
    uint64_t h = fnv1a(s, n);
    for (uint32_t i = (uint32_t)(h & (HASH_SLOTS - 1)); ; i = (i + 1) & (HASH_SLOTS - 1)) {
        if (tk_slot[i].rank < 0) {
            tk_slot[i].off = arena_off; tk_slot[i].len = (uint16_t)n;
            tk_slot[i].rank = rank;
            return;
        }
    }
}

static const int B64[256] = {
    ['A']=0,['B']=1,['C']=2,['D']=3,['E']=4,['F']=5,['G']=6,['H']=7,['I']=8,
    ['J']=9,['K']=10,['L']=11,['M']=12,['N']=13,['O']=14,['P']=15,['Q']=16,
    ['R']=17,['S']=18,['T']=19,['U']=20,['V']=21,['W']=22,['X']=23,['Y']=24,
    ['Z']=25,['a']=26,['b']=27,['c']=28,['d']=29,['e']=30,['f']=31,['g']=32,
    ['h']=33,['i']=34,['j']=35,['k']=36,['l']=37,['m']=38,['n']=39,['o']=40,
    ['p']=41,['q']=42,['r']=43,['s']=44,['t']=45,['u']=46,['v']=47,['w']=48,
    ['x']=49,['y']=50,['z']=51,['0']=52,['1']=53,['2']=54,['3']=55,['4']=56,
    ['5']=57,['6']=58,['7']=59,['8']=60,['9']=61,['+']=62,['/']=63,
};

static void tk_load(const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/o200k_base.tiktoken", dir);
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open o200k_base.tiktoken");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) die("short read on tiktoken file");
    buf[sz] = 0; fclose(f);
    tk_arena = malloc(sz);  /* decoded tokens are smaller than base64 text */
    for (uint32_t i = 0; i < HASH_SLOTS; i++) tk_slot[i].rank = -1;
    uint32_t apos = 0;
    for (char *line = buf; line < buf + sz; ) {
        char *sp = strchr(line, ' ');
        char *nl = strchr(line, '\n');
        if (!sp || !nl) break;
        uint32_t start = apos;
        uint32_t acc = 0; int nb = 0;
        for (char *q = line; q < sp; q++) {
            if (*q == '=') break;
            acc = (acc << 6) | (uint32_t)B64[(uint8_t)*q];
            if (++nb == 4) {
                tk_arena[apos++] = acc >> 16;
                tk_arena[apos++] = (acc >> 8) & 0xFF;
                tk_arena[apos++] = acc & 0xFF;
                acc = 0; nb = 0;
            }
        }
        if (nb == 3) { tk_arena[apos++] = acc >> 10; tk_arena[apos++] = (acc >> 2) & 0xFF; }
        else if (nb == 2) { tk_arena[apos++] = acc >> 4; }
        int32_t rank = (int32_t)strtol(sp + 1, NULL, 10);
        int len = (int)(apos - start);
        if (len > tk_max_len) tk_max_len = len;
        tk_insert(tk_arena + start, apos - start, rank, start);
        line = nl + 1;
    }
    free(buf);
}

/* --- o200k_base pretokenizer: hand-compiled scanner for
 * [^\r\nLN]?[U]*[D]+('s..)?|[^\r\nLN]?[U]+[D]*('s..)?|N{1,3}| ?[^\sLN]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
 */
typedef struct { uint32_t cp; int len, flags; } Ch;
static inline Ch peek(const uint8_t *s, size_t i, size_t n) {
    Ch c = {0, 0, 0};
    if (i >= n) return c;
    c.len = utf8_dec(s + i, n - i, &c.cp);
    c.flags = uc_flags(c.cp);
    return c;
}

/* optional (?i:'s|'t|'re|'ve|'m|'ll|'d) */
static size_t suffix_len(const uint8_t *s, size_t i, size_t n) {
    if (i >= n || s[i] != '\'') return 0;
    if (i + 1 < n) {
        uint8_t a = s[i + 1] | 0x20;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') {
            if (a == 't' || a == 'r' || a == 'v' || a == 'l') { /* unreachable for t */ }
            /* 're / 've / 'll take priority handling below */
        }
        uint8_t b = (i + 2 < n) ? (s[i + 2] | 0x20) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
            return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    return 0;
}

/* word alt: U*D+ (v=1) or U+D* (v=2) starting at i; returns end or 0 */
static size_t word_body(const uint8_t *s, size_t i, size_t n, int v) {
    size_t ustart = i;
    size_t upos[512]; int nup = 0;          /* codepoint boundaries in U-run */
    while (i < n && nup < 511) {
        Ch c = peek(s, i, n);
        if (!(c.flags & FL_U)) break;
        upos[nup++] = i;
        i += c.len;
    }
    size_t uend = i;
    if (v == 2) {                            /* U+D*: need >=1 U, then greedy D */
        if (nup == 0) return 0;
        while (i < n) {
            Ch c = peek(s, i, n);
            if (!(c.flags & FL_D)) break;
            i += c.len;
        }
        return i + suffix_len(s, i, n) - 0;
    }
    /* v==1: U*D+ with backtracking: largest q in [ustart..uend] with s[q] in D */
    size_t q = uend; int found = 0;
    Ch c = peek(s, q, n);
    if (q < n && (c.flags & FL_D)) found = 1;
    if (!found) {
        for (int k = nup - 1; k >= 0; k--) {
            c = peek(s, upos[k], n);
            if (c.flags & FL_D) { q = upos[k]; found = 1; break; }
        }
    }
    if (!found) return 0;
    (void)ustart;
    i = q;
    while (i < n) {
        c = peek(s, i, n);
        if (!(c.flags & FL_D)) break;
        i += c.len;
    }
    return i + suffix_len(s, i, n);
}

/* next pretoken starting at i; returns end (>i) */
static size_t pretok_next(const uint8_t *s, size_t i, size_t n) {
    Ch c0 = peek(s, i, n);
    int pfx_ok = c0.cp != '\r' && c0.cp != '\n' && !(c0.flags & (FL_L | FL_N));
    size_t e;
    /* alt1: [^\r\nLN]? U* D+ sfx? — with prefix, then without */
    if (pfx_ok && (e = word_body(s, i + c0.len, n, 1)) > 0) return e;
    if ((e = word_body(s, i, n, 1)) > i) return e;
    /* alt2: [^\r\nLN]? U+ D* sfx? */
    if (pfx_ok && (e = word_body(s, i + c0.len, n, 2)) > 0) return e;
    if ((e = word_body(s, i, n, 2)) > i) return e;
    /* alt3: N{1,3} */
    if (c0.flags & FL_N) {
        size_t p = i;
        for (int k = 0; k < 3 && p < n; k++) {
            Ch c = peek(s, p, n);
            if (!(c.flags & FL_N)) break;
            p += c.len;
        }
        return p;
    }
    /* alt4: " "? [^\s L N]+ [\r\n/]* */
    {
        size_t p = i;
        if (s[p] == ' ') p++;
        size_t run = p;
        while (run < n) {
            Ch c = peek(s, run, n);
            if ((c.flags & (FL_WS | FL_L | FL_N))) break;
            run += c.len;
        }
        if (run > p) {
            while (run < n && (s[run] == '\r' || s[run] == '\n' || s[run] == '/')) run++;
            return run;
        }
    }
    /* whitespace alts: maximal \s run */
    if (c0.flags & FL_WS) {
        size_t j = i, last_nl_end = 0, nws = 0, prev = i;
        while (j < n) {
            Ch c = peek(s, j, n);
            if (!(c.flags & FL_WS)) break;
            prev = j;
            j += c.len;
            nws++;
            if (c.cp == '\r' || c.cp == '\n') last_nl_end = j;
        }
        if (last_nl_end > 0) return last_nl_end;          /* alt5: \s*[\r\n]+ */
        if (j >= n) return j;                             /* alt6 at EOT      */
        if (nws >= 2) return prev;                        /* alt6: drop last  */
        return j;                                         /* alt7: \s+        */
    }
    /* fallback: single (invalid?) byte/codepoint */
    return i + (c0.len > 0 ? (size_t)c0.len : 1);
}

/* BPE-encode s[a..b) appending token ids + byte ranges */
typedef struct { int32_t *id; uint32_t *bs, *be; size_t n, cap; } TokBuf;
static void tok_push(TokBuf *t, int32_t id, uint32_t bs, uint32_t be) {
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 4096;
        t->id = realloc(t->id, t->cap * 4);
        t->bs = realloc(t->bs, t->cap * 4);
        t->be = realloc(t->be, t->cap * 4);
    }
    t->id[t->n] = id; t->bs[t->n] = bs; t->be[t->n] = be; t->n++;
}

static void bpe_chunk(const uint8_t *s, size_t a, size_t b, TokBuf *out) {
    size_t n = b - a;
    int32_t whole = tk_rank(s + a, n);
    if (whole >= 0) { tok_push(out, whole, a, b); return; }
    /* boundaries[i] = start of part i; parts are byte ranges */
    uint32_t *bd = malloc((n + 1) * 4);
    int32_t *pr = malloc(n * 4);            /* pr[i] = rank of parts i+i+1 merged */
    size_t np = n;
    for (size_t i = 0; i <= n; i++) bd[i] = (uint32_t)(a + i);
    for (size_t i = 0; i + 1 < np; i++)
        pr[i] = tk_rank(s + bd[i], bd[i + 2] - bd[i]);
    while (np > 1) {
        int32_t best = INT32_MAX; size_t bi = SIZE_MAX;
        for (size_t i = 0; i + 1 < np; i++)
            if (pr[i] >= 0 && pr[i] < best) { best = pr[i]; bi = i; }
        if (bi == SIZE_MAX) break;
        /* merge parts bi,bi+1 */
        memmove(bd + bi + 1, bd + bi + 2, (np - bi - 1) * 4);
        memmove(pr + bi, pr + bi + 1, (np - bi - 2 + 1) * 4);
        np--;
        if (bi > 0)      pr[bi - 1] = tk_rank(s + bd[bi - 1], bd[bi + 1] - bd[bi - 1]);
        if (bi + 1 < np) pr[bi] = tk_rank(s + bd[bi], bd[bi + 2] - bd[bi]);
    }
    for (size_t i = 0; i < np; i++) {
        int32_t r = tk_rank(s + bd[i], bd[i + 1] - bd[i]);
        if (r < 0) die("BPE: unknown token piece");
        tok_push(out, r, bd[i], bd[i + 1]);
    }
    free(bd); free(pr);
}

static const struct { const char *lit; int32_t id; } SPECIALS[] = {
    {"<|endoftext|>", 199999}, {"<|endofprompt|>", 200018},
};

static void tokenize(const uint8_t *s, size_t n, TokBuf *out) {
    size_t i = 0;
    while (i < n) {
        /* allowed_special="all": match special tokens first */
        size_t sp_end = 0; int32_t sp_id = -1;
        size_t next_sp = n;
        for (size_t k = 0; k < sizeof SPECIALS / sizeof *SPECIALS; k++) {
            const char *lit = SPECIALS[k].lit;
            size_t ll = strlen(lit);
            const uint8_t *hit = memmem(s + i, n - i, lit, ll);
            if (hit && (size_t)(hit - s) < next_sp) {
                next_sp = hit - s;
                sp_end = next_sp + ll;
                sp_id = SPECIALS[k].id;
            }
        }
        /* pretokenize + BPE up to the next special (or end) */
        size_t stop = next_sp;
        while (i < stop) {
            size_t e = pretok_next(s, i, stop);
            if (e > stop) e = stop;
            bpe_chunk(s, i, e, out);
            i = e;
        }
        if (sp_id >= 0) {
            tok_push(out, sp_id, next_sp, sp_end);
            i = sp_end;
        }
    }
}

/* ---------------------------------------------------------------- forward */
static struct {
    int T;                 /* window tokens */
    float *x;              /* [T][D] residual  */
    float *tb;             /* [T][D] normed    */
    bf16  *tbh;            /* [T][D] normed, bf16 */
    int8_t *xq; float *xs; /* --int8: [T][D] tb quantized per token, [T] scales */
    float *q;              /* [T][QD]          */
    float *k, *v;          /* [T][KVD]         */
    float *ao;             /* [T][QD]          */
    float *kt; int ktld;   /* SME: [KVD][ktld] k transposed (ph_a_sme), ktld = T + KW */
    float *rc, *rs;        /* [T][HD/2] rope   */
    float *gates;          /* [T][TOPK]        */
    int   *eidx;           /* [T][TOPK]        */
    float *moe;            /* [T][TOPK][D]     */
    float *lp;             /* [T][NC] logprobs */
    int   *bucket;         /* [NE] heads into chain */
    int   *chain;          /* [T*TOPK] next    */
    int   *wlist;          /* [T*TOPK] items grouped by expert */
    struct WChunk { int e, off, cnt; } *wchunk;
    int   nchunk;
    int   *pos, *sst, *sen; /* [T] rope position, segment start/end (packed batching) */
    const int32_t *ids;
    int layer;
} R;

static void buf_alloc(int T) {
    R.x  = malloc((size_t)T * D * 4);   R.tb = malloc((size_t)T * D * 4);
    R.tbh = malloc((size_t)T * D * 2);
    R.q  = malloc((size_t)T * QD * 4);  R.ao = malloc((size_t)T * QD * 4);
    R.k  = malloc((size_t)T * KVD * 4); R.v  = malloc((size_t)T * KVD * 4);
    R.rc = malloc((size_t)T * HD / 2 * 4); R.rs = malloc((size_t)T * HD / 2 * 4);
    R.gates = malloc((size_t)T * TOPK * 4); R.eidx = malloc((size_t)T * TOPK * 4);
    R.moe = malloc((size_t)T * TOPK * D * 4);
    if (g_sme) {           /* zero tail: attention reads up to KW past the batch end */
        R.ktld = T + KW;
        if (!(R.kt = calloc((size_t)KVD * R.ktld, 4))) die("out of memory");
    }
    if (g_q8 && (!(R.xq = malloc((size_t)T * D)) || !(R.xs = malloc((size_t)T * 4))))
        die("out of memory");
    R.lp  = malloc((size_t)T * NC * 4);
    R.bucket = malloc(NE * 4); R.chain = malloc((size_t)T * TOPK * 4);
    R.wlist = malloc((size_t)T * TOPK * 4);
    R.wchunk = malloc(((size_t)T * TOPK / MOE_B + NE + 1) * sizeof *R.wchunk);
    R.pos = malloc((size_t)T * 4);
    R.sst = malloc((size_t)T * 4);
    R.sen = malloc((size_t)T * 4);
}

static void rope_init(int T) {
    /* YaRN (see reference RotaryEmbedding) */
    double conc = 0.1 * log(YARN_FACTOR) + 1.0;
    double inv[HD / 2];
    double d_half = HD / 2.0;
    double low  = d_half * log(YARN_ORIG / (YARN_BETA  * 2 * M_PI)) / log(ROPE_THETA);
    double high = d_half * log(YARN_ORIG / (YARN_ALPHA * 2 * M_PI)) / log(ROPE_THETA);
    for (int i = 0; i < HD / 2; i++) {
        double freq = pow(ROPE_THETA, (2.0 * i) / HD);
        double interp = 1.0 / (YARN_FACTOR * freq), extrap = 1.0 / freq;
        double ramp = (i - low) / (high - low);
        double mask = 1.0 - fmin(fmax(ramp, 0.0), 1.0);
        inv[i] = interp * (1 - mask) + extrap * mask;
    }
    for (int t = 0; t < T; t++)
        for (int i = 0; i < HD / 2; i++) {
            R.rc[t * (HD / 2) + i] = (float)(cos(t * inv[i]) * conc);
            R.rs[t * (HD / 2) + i] = (float)(sin(t * inv[i]) * conc);
        }
}

static void rms_norm(const float *x, const bf16 *scale, float *y) {
    float ss = dot_f32(x, x, D);
    float r = 1.0f / sqrtf(ss / D + RMS_EPS);
    for (int i = 0; i < D; i++) y[i] = x[i] * r * b2f(scale[i]);
}

static void rope_apply(float *vec, int nheads, const float *rc, const float *rs) {
    for (int h = 0; h < nheads; h++) {
        float *p = vec + h * HD;
        for (int i = 0; i < HD / 2; i++) {
            float x1 = p[2 * i], x2 = p[2 * i + 1];
            p[2 * i]     = x1 * rc[i] - x2 * rs[i];
            p[2 * i + 1] = x2 * rc[i] + x1 * rs[i];
        }
    }
}

/* phase A: moe-reduce (layer>0) + attn norm + qkv + rope */
static void a_pre(size_t t) {      /* residual + norm -> R.tb / R.tbh */
    Layer *y = &M.L[R.layer];
    float *x = R.x + t * D;
    if (R.layer > 0) {
        const float *m = R.moe + t * TOPK * D;
        for (int s = 0; s < TOPK; s++)
            for (int i = 0; i < D; i++) x[i] += m[s * D + i];
    }
    float *tb = R.tb + t * D;
    rms_norm(x, y->attn_norm, tb);
    CVT_H(tb, R.tbh + t * D, D);
}
static void a_post(size_t t) {     /* rope + score scale on projected q, k */
    float *q = R.q + t * QD, *k = R.k + t * KVD;
    const float *rc = R.rc + (size_t)R.pos[t] * (HD / 2);
    const float *rs = R.rs + (size_t)R.pos[t] * (HD / 2);
    rope_apply(q, NH, rc, rs);
    rope_apply(k, NKV, rc, rs);
    for (int i = 0; i < QD; i++)  q[i] *= QK_SCALE;
    for (int i = 0; i < KVD; i++) k[i] *= QK_SCALE;
}
static void ph_a(void *ctx, size_t t) {
    (void)ctx;
    Layer *y = &M.L[R.layer];
    a_pre(t);
    const float *tb = R.tb + t * D;
    const bf16 *th = R.tbh + t * D;
    (void)tb; (void)th;            /* DOT_X reads one of them, depending on the build */
    float *q = R.q + t * QD, *k = R.k + t * KVD, *v = R.v + t * KVD;
    for (int o = 0; o < QD; o++)
        q[o] = DOT_X(tb, th, y->qkv_w + (size_t)o * D, D) + b2f(y->qkv_b[o]);
    for (int o = 0; o < KVD; o++)
        k[o] = DOT_X(tb, th, y->qkv_w + (size_t)(QD + o) * D, D) + b2f(y->qkv_b[QD + o]);
    for (int o = 0; o < KVD; o++)
        v[o] = DOT_X(tb, th, y->qkv_w + (size_t)(QD + KVD + o) * D, D)
               + b2f(y->qkv_b[QD + KVD + o]);
    a_post(t);
}
#if defined(HAVE_SME)
static void ph_a_sme(void *ctx, size_t b) {
    (void)ctx;
    Layer *y = &M.L[R.layer];
    const size_t t0 = b * DB;
    const int m = R.T - (int)t0 < DB ? R.T - (int)t0 : DB;
    bf16 xp[DB * D];
    float out[DB * QKVD];
    const bf16 *rows[DB] = {0};
    for (int i = 0; i < m; i++) { a_pre(t0 + i); rows[i] = R.tbh + (t0 + i) * D; }
    pack_rows(rows, m, DB, D, xp);
    sme_gemm(xp, DB, y->qkv_p, out, D, QKVD);
    for (int i = 0; i < m; i++) {
        const size_t t = t0 + i;
        const float *r = out + (size_t)i * QKVD;
        float *q = R.q + t * QD, *k = R.k + t * KVD, *v = R.v + t * KVD;
        for (int o = 0; o < QD; o++)  q[o] = r[o] + b2f(y->qkv_b[o]);
        for (int o = 0; o < KVD; o++) k[o] = r[QD + o] + b2f(y->qkv_b[QD + o]);
        for (int o = 0; o < KVD; o++) v[o] = r[QD + KVD + o] + b2f(y->qkv_b[QD + KVD + o]);
        a_post(t);
        for (int c = 0; c < KVD; c++) R.kt[(size_t)c * R.ktld + t] = k[c];   /* for attn_sme */
    }
}
#endif

/* phase B: banded attention + out proj + residual + mlp norm + router */
static void b_attn(size_t t) {     /* banded attention -> R.ao */
    int j0 = (int)t - BAND; if (j0 < R.sst[t]) j0 = R.sst[t];
    int j1 = (int)t + BAND; if (j1 >= R.sen[t]) j1 = R.sen[t] - 1;
    float *ao = R.ao + t * QD;
    float sc[2 * BAND + 1];
    for (int h = 0; h < NH; h++) {
        const float *qh = R.q + t * QD + h * HD;
        int g = h / (NH / NKV);
        float sink = M.L[R.layer].sinks[h] * (float)M_LN2;
        float mx = sink;
        for (int j = j0; j <= j1; j++) {
            float s = dot_f32(qh, R.k + (size_t)j * KVD + g * HD, HD);
            sc[j - j0] = s;
            if (s > mx) mx = s;
        }
        float den = expf(sink - mx);
        int W = j1 - j0 + 1;
        float *o = ao + h * HD;
#if defined(__aarch64__)
        {   /* vectorized exp over the window, then V-accumulation with the
             * 64-float output held in 16 NEON registers across the sweep */
            float32x4_t vden = vdupq_n_f32(0), vmx = vdupq_n_f32(mx);
            int j = 0;
            for (; j + 4 <= W; j += 4) {
                float32x4_t e = exp4(vsubq_f32(vld1q_f32(sc + j), vmx));
                vst1q_f32(sc + j, e);
                vden = vaddq_f32(vden, e);
            }
            den += vaddvq_f32(vden);
            for (; j < W; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
            float32x4_t acc[HD / 4];
            for (int i = 0; i < HD / 4; i++) acc[i] = vdupq_n_f32(0);
            const float *vj = R.v + (size_t)j0 * KVD + g * HD;
            for (j = 0; j < W; j++, vj += KVD) {
                float32x4_t wv = vdupq_n_f32(sc[j]);
                for (int i = 0; i < HD / 4; i++)
                    acc[i] = vfmaq_f32(acc[i], wv, vld1q_f32(vj + 4 * i));
            }
            float32x4_t inv = vdupq_n_f32(1.0f / den);
            for (int i = 0; i < HD / 4; i++)
                vst1q_f32(o + 4 * i, vmulq_f32(acc[i], inv));
        }
#else
        for (int j = 0; j < W; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
        for (int i = 0; i < HD; i++) o[i] = 0;
        for (int j = 0; j < W; j++) {
            float w = sc[j] / den;
            const float *vj = R.v + (size_t)(j0 + j) * KVD + g * HD;
            for (int i = 0; i < HD; i++) o[i] += w * vj[i];
        }
#endif
    }
}
static void b_norm(size_t t) {     /* mlp norm (+ --int8 rotated quantized copy) */
    Layer *y = &M.L[R.layer];
    float *tb = R.tb + t * D;
    rms_norm(R.x + t * D, y->mlp_norm, tb);
    CVT_H(tb, R.tbh + t * D, D);
    if (g_q8 && R.layer >= Q8_FIRST) {   /* rotated copy: the router reads tb as is */
        float xr[D];
        memcpy(xr, tb, sizeof xr);
        fwht_row(xr, D);
        q8_row(xr, D, R.xq + t * D, R.xs + t);
    }
}
static void b_route(size_t t, float *g) {   /* router logits g[NE] -> top-k gates */
    int   *ei = R.eidx + t * TOPK;
    float *gw = R.gates + t * TOPK;
    for (int s = 0; s < TOPK; s++) {
        int bi = 0; float bv = -1e30f;
        for (int e = 0; e < NE; e++)
            if (g[e] > bv) { bv = g[e]; bi = e; }
        ei[s] = bi; gw[s] = bv; g[bi] = -1e30f;
    }
    float mx = gw[0], den = 0;
    for (int s = 0; s < TOPK; s++) { gw[s] = expf(gw[s] - mx); den += gw[s]; }
    for (int s = 0; s < TOPK; s++) gw[s] /= den;
    if (g_skip_beta > 0.0f) {
        /* dynamic expert skipping (Lu et al. 2024): drop slot s when its
         * routing weight is far below the top expert's, renormalize the rest */
        float thr = g_skip_beta * gw[0], kden = 0;
        int keep = TOPK;
        for (int s = 0; s < TOPK; s++) {
            if (s > 0 && gw[s] < thr) { keep = s; break; }
            kden += gw[s];
        }
        for (int s = keep; s < TOPK; s++) {
            ei[s] = -1;
            memset(R.moe + ((size_t)t * TOPK + s) * D, 0, D * 4);
        }
        for (int s = 0; s < keep; s++) gw[s] /= kden;
    }
}
static void b_post(size_t t) {     /* mlp norm + router + top-k gates */
    Layer *y = &M.L[R.layer];
    const float *tb = R.tb + t * D;
    const bf16 *th = R.tbh + t * D;
    (void)tb; (void)th;            /* DOT_X reads one of them, depending on the build */
    b_norm(t);
    float g[NE];
    for (int e = 0; e < NE; e++)
        g[e] = DOT_X(tb, th, y->gate_w + (size_t)e * D, D) + b2f(y->gate_b[e]);
    b_route(t, g);
}
static void ph_b(void *ctx, size_t t) {
    (void)ctx;
    Layer *y = &M.L[R.layer];
    b_attn(t);
    const float *ao = R.ao + t * QD;
    float *x = R.x + t * D;
    bf16 aoh[QD];
    CVT_H(ao, aoh, QD);
    for (int o = 0; o < D; o++)
        x[o] += DOT_X(ao, aoh, y->attn_out_w + (size_t)o * QD, QD) + b2f(y->attn_out_b[o]);
    b_post(t);
}
#if defined(HAVE_SME)
/* SME attention for m <= DB query tokens t0..: per head, the block's union key
 * window [kmin, kmax] (packed segments may straddle a block) is scored as
 * S^T = K Q^T, one row per key and one lane per query token, so the masked
 * softmax (band + segment) vectorizes across tokens on NEON (streaming mode
 * has no NEON) and leaves P^T in the layout O = P V consumes. K^T comes
 * pre-transposed from ph_a_sme. fp32 throughout. */
static void attn_sme(size_t t0, int m) {
    int32_t lo[DB], hi[DB];
    int kmin = R.T, kmax = -1;
    for (int i = 0; i < DB; i++) {
        if (i >= m) { lo[i] = INT32_MAX; hi[i] = INT32_MIN; continue; }   /* no keys */
        const int t = (int)t0 + i;
        int j0 = t - BAND, j1 = t + BAND;
        if (j0 < R.sst[t]) j0 = R.sst[t];
        if (j1 >= R.sen[t]) j1 = R.sen[t] - 1;
        lo[i] = j0; hi[i] = j1;
        if (j0 < kmin) kmin = j0;
        if (j1 > kmax) kmax = j1;
    }
    const int nk = kmax - kmin + 1, nkp = (nk + 15) & ~15;
    float qt[HD * DB], st[KW * DB], o[DB * HD], dn[DB];
    const float *sinks = M.L[R.layer].sinks;
    int32x4_t vlo[DB / 4], vhi[DB / 4];
    for (int q = 0; q < DB / 4; q++) { vlo[q] = vld1q_s32(lo + 4 * q); vhi[q] = vld1q_s32(hi + 4 * q); }
    const float32x4_t ninf = vdupq_n_f32(-INFINITY), zero = vdupq_n_f32(0);
    for (int h = 0; h < NH; h++) {
        const int g = h / (NH / NKV);
        for (int d = 0; d < HD; d++)
            for (int i = 0; i < DB; i++)
                qt[d * DB + i] = i < m ? R.q[(t0 + i) * QD + h * HD + d] : 0.0f;
        sme_gemm_f32(R.kt + (size_t)g * HD * R.ktld + kmin, R.ktld, nkp, qt, DB, st, DB, HD, DB);
        const float32x4_t sink = vdupq_n_f32(sinks[h] * (float)M_LN2);
        float32x4_t mx[DB / 4], den[DB / 4];
        for (int q = 0; q < DB / 4; q++) mx[q] = sink;
        for (int j = 0; j < nk; j++) {
            const int32x4_t kj = vdupq_n_s32(kmin + j);
            for (int q = 0; q < DB / 4; q++) {
                uint32x4_t ok = vandq_u32(vcgeq_s32(kj, vlo[q]), vcleq_s32(kj, vhi[q]));
                mx[q] = vmaxq_f32(mx[q], vbslq_f32(ok, vld1q_f32(st + j * DB + 4 * q), ninf));
            }
        }
        for (int q = 0; q < DB / 4; q++) den[q] = exp4(vsubq_f32(sink, mx[q]));
        for (int j = 0; j < nk; j++) {
            const int32x4_t kj = vdupq_n_s32(kmin + j);
            for (int q = 0; q < DB / 4; q++) {
                uint32x4_t ok = vandq_u32(vcgeq_s32(kj, vlo[q]), vcleq_s32(kj, vhi[q]));
                float *sp = st + j * DB + 4 * q;
                float32x4_t e = vbslq_f32(ok, exp4(vsubq_f32(vld1q_f32(sp), mx[q])), zero);
                vst1q_f32(sp, e);
                den[q] = vaddq_f32(den[q], e);
            }
        }
        for (int q = 0; q < DB / 4; q++) vst1q_f32(dn + 4 * q, den[q]);
        sme_gemm_f32(st, DB, DB, R.v + (size_t)kmin * KVD + g * HD, KVD, o, HD, nk, HD);
        for (int i = 0; i < m; i++) {
            const float inv = 1.0f / dn[i];
            float *dst = R.ao + (t0 + i) * QD + h * HD;
            for (int d = 0; d < HD; d++) dst[d] = o[i * HD + d] * inv;
        }
    }
}

static void ph_b_sme(void *ctx, size_t b) {
    (void)ctx;
    Layer *y = &M.L[R.layer];
    const size_t t0 = b * DB;
    const int m = R.T - (int)t0 < DB ? R.T - (int)t0 : DB;
    bf16 xp[DB * QD], hb[DB][QD];
    float out[DB * D];
    const bf16 *rows[DB] = {0};
    attn_sme(t0, m);
    for (int i = 0; i < m; i++) {
        cvt_f32_bf16(R.ao + (t0 + i) * QD, hb[i], QD);
        rows[i] = hb[i];
    }
    pack_rows(rows, m, DB, QD, xp);
    sme_gemm(xp, DB, y->out_p, out, QD, D);
    for (int i = 0; i < m; i++) {
        float *x = R.x + (t0 + i) * D;
        const float *r = out + (size_t)i * D;
        for (int o = 0; o < D; o++) x[o] += r[o] + b2f(y->attn_out_b[o]);
        b_norm(t0 + i);
        rows[i] = R.tbh + (t0 + i) * D;
    }
    pack_rows(rows, m, DB, D, xp);  /* router: [DB][D] x [D][NE] */
    sme_gemm(xp, DB, y->gate_p, out, D, NE);
    for (int i = 0; i < m; i++) {
        float *g = out + (size_t)i * NE;
        for (int e = 0; e < NE; e++) g[e] += b2f(y->gate_b[e]);
        b_route(t0 + i, g);
    }
}
#endif

/* swiglu halves -> h (chunked layout, gpt-oss clamps and +1 linear bias) */
static void swiglu_act(const float *h1, float *h, int n) {
#if defined(__aarch64__)
    const float32x4_t lim = vdupq_n_f32(SWIGLU_LIMIT), nlim = vdupq_n_f32(-SWIGLU_LIMIT);
    const float32x4_t one = vdupq_n_f32(1.0f), nalpha = vdupq_n_f32(-SWIGLU_ALPHA);
    for (int i = 0; i < n; i += 4) {
        float32x4_t glu = vminq_f32(vld1q_f32(h1 + i), lim);
        float32x4_t lin = vminq_f32(vmaxq_f32(vld1q_f32(h1 + n + i), nlim), lim);
        float32x4_t sig = vdivq_f32(one, vaddq_f32(one, exp4(vmulq_f32(nalpha, glu))));
        vst1q_f32(h + i, vmulq_f32(vmulq_f32(glu, sig), vaddq_f32(lin, one)));
    }
#else
    for (int i = 0; i < n; i++) {
        float glu = h1[i], lin = h1[n + i];
        if (glu > SWIGLU_LIMIT) glu = SWIGLU_LIMIT;
        if (lin > SWIGLU_LIMIT) lin = SWIGLU_LIMIT;
        if (lin < -SWIGLU_LIMIT) lin = -SWIGLU_LIMIT;
        h[i] = glu / (1.0f + expf(-SWIGLU_ALPHA * glu)) * (lin + 1.0f);
    }
#endif
}

/* --calib: fold one chunk's per-neuron importance into the global totals */
static void calib_add(int e, const double *imp) {
    pthread_mutex_lock(&g_calib_mu);
    double *c = g_calib[R.layer][e];
    for (int j = 0; j < FF; j++) c[j] += imp[j];
    pthread_mutex_unlock(&g_calib_mu);
}

/* phase C: experts (parallel over experts, chained token lists) */
#define MOE_BP ((MOE_B + 15) & ~15)   /* expert block rows, padded to the 16-row tile */
/* --int8 expert block: m <= MOE_B (token, slot) items of expert e */
static void moe_block_q8(int e, const int *blk, int m) {
    const int l = R.layer, W = g_ff, MP = (m + 15) & ~15;
    const Layer *y = &M.L[l];
    const int8_t *w1 = g_w8[l][0] + (size_t)e * D * 2 * W, *w2 = g_w8[l][1] + (size_t)e * W * D;
    const float *s1 = g_s8[l][0] + (size_t)e * 2 * W, *s2 = g_s8[l][1] + (size_t)e * D;
    int32_t acc[MOE_BP * 2 * FF];
    int8_t xp[MOE_BP * D] __attribute__((aligned(64)));   /* D == FF: either GEMM's input */
    int8_t hq[FF];
    float xs[MOE_BP], b1[2 * FF], b2[D], f[2 * FF], h[FF];
    for (int i = 0; i < MP; i++) {     /* gather [D/4][MP][4]; pad rows zero */
        const int8_t *src = i < m ? R.xq + (size_t)(blk[i] / TOPK) * D : NULL;
        if (src) xs[i] = R.xs[blk[i] / TOPK];
        for (int k = 0; k < D / 4; k++) {
            if (src) memcpy(xp + ((size_t)k * MP + i) * 4, src + 4 * k, 4);
            else memset(xp + ((size_t)k * MP + i) * 4, 0, 4);
        }
    }
    gemm_s8(xp, MP, m, w1, acc, D, 2 * W);
    cvt_bf16(y->b1 + (size_t)e * 2 * W, b1, 2 * W);
    cvt_bf16(y->b2 + (size_t)e * D, b2, D);
    for (int i = 0; i < m; i++) {
        const int32_t *r = acc + (size_t)i * 2 * W;
        for (int j = 0; j < 2 * W; j++) f[j] = (float)r[j] * (xs[i] * s1[j]) + b1[j];
        swiglu_act(f, h, W);
        fwht_row(h, W);
        q8_row(h, W, hq, &xs[i]);      /* [W/4][MP][4]; pad rows still zero */
        for (int k = 0; k < W / 4; k++) memcpy(xp + ((size_t)k * MP + i) * 4, hq + 4 * k, 4);
    }
    gemm_s8(xp, MP, m, w2, acc, W, D);
    for (int i = 0; i < m; i++) {
        const float gw = R.gates[blk[i]];
        float *o = R.moe + (size_t)blk[i] * D;
        const int32_t *r = acc + (size_t)i * D;
        for (int j = 0; j < D; j++) o[j] = ((float)r[j] * (xs[i] * s2[j]) + b2[j]) * gw;
    }
}
static void ph_c_q8(void *ctx, size_t c) {
    (void)ctx;
    moe_block_q8(R.wchunk[c].e, R.wlist + R.wchunk[c].off, R.wchunk[c].cnt);
}

#if defined(HAVE_SME)
static void moe_block_sme(const bf16 *w1, const bf16 *b1, const bf16 *w2, const bf16 *b2,
                      const int *blk, int m, int e) {
    const int W = g_ff, MP = (m + 15) & ~15;
    float y1[MOE_BP * 2 * FF], y2[MOE_BP * D], bias[2 * FF], h[FF];
    bf16 xp[MOE_BP * FF], hb[MOE_BP][FF];    /* D == FF: xp fits either GEMM's input */
    double imp[FF];
    const bf16 *rows[MOE_BP] = {0};    /* only [0, m) read; init quiets gcc */
    if (g_calib) memset(imp, 0, sizeof imp);
    for (int i = 0; i < m; i++) rows[i] = R.tbh + (size_t)(blk[i] / TOPK) * D;
    pack_rows(rows, m, MP, D, xp);
    sme_gemm(xp, MP, w1, y1, D, 2 * W);
    cvt_bf16(b1, bias, 2 * W);
    for (int i = 0; i < m; i++) {
        float *r = y1 + (size_t)i * 2 * W;
        for (int j = 0; j < 2 * W; j++) r[j] += bias[j];
        swiglu_act(r, h, W);
        if (g_calib) {
            float gw = R.gates[blk[i]];
            for (int j = 0; j < W; j++) imp[j] += gw * fabsf(h[j]);
        }
        cvt_f32_bf16(h, hb[i], W);
        rows[i] = hb[i];
    }
    pack_rows(rows, m, MP, W, xp);
    sme_gemm(xp, MP, w2, y2, W, D);
    cvt_bf16(b2, bias, D);
    for (int i = 0; i < m; i++) {
        float gw = R.gates[blk[i]];
        float *o = R.moe + (size_t)blk[i] * D;
        const float *r = y2 + (size_t)i * D;
        for (int j = 0; j < D; j++) o[j] = (r[j] + bias[j]) * gw;
    }
    if (g_calib) calib_add(e, imp);
}
#endif
#if defined(HAVE_BFDOT)
static void moe_block(const bf16 *w1, const bf16 *b1, const bf16 *w2, const bf16 *b2,
                      const int *blk, int m, int e) {
    const int W = g_ff;
    float h1s[MOE_B][2 * FF], outs[MOE_B][D], h[FF];
    bf16 hhs[MOE_B][FF];
    double imp[FF];
    const bf16 *xh[MOE_B] = {0}; float *yp[MOE_B] = {0};   /* only [0, m) read; init quiets gcc */
    if (g_calib) memset(imp, 0, sizeof imp);
    for (int i = 0; i < m; i++) {
        xh[i] = R.tbh + (size_t)(blk[i] / TOPK) * D;
        yp[i] = h1s[i];
        cvt_bf16(b1, h1s[i], 2 * W);
    }
    moe_gemm(w1, xh, yp, m, D, 2 * W);
    for (int i = 0; i < m; i++) {
        swiglu_act(h1s[i], h, W);
        if (g_calib) {
            float gw = R.gates[blk[i]];
            for (int j = 0; j < W; j++) imp[j] += gw * fabsf(h[j]);
        }
        cvt_f32_bf16(h, hhs[i], W);
        cvt_bf16(b2, outs[i], D);
        xh[i] = hhs[i]; yp[i] = outs[i];
    }
    moe_gemm(w2, xh, yp, m, W, D);
    for (int i = 0; i < m; i++) {
        float gw = R.gates[blk[i]];
        float *o = R.moe + (size_t)blk[i] * D;
        for (int j = 0; j < D; j++) o[j] = outs[i][j] * gw;
    }
    if (g_calib) calib_add(e, imp);
}
#endif

#if defined(HAVE_BFDOT)
static void ph_c(void *ctx, size_t c) {
    (void)ctx;
    Layer *y = &M.L[R.layer];
    size_t e = (size_t)R.wchunk[c].e, W = (size_t)g_ff;
#if defined(HAVE_SME)
    void (*blk)(const bf16 *, const bf16 *, const bf16 *, const bf16 *, const int *, int, int) =
        g_sme ? moe_block_sme : moe_block;
#else
    void (*blk)(const bf16 *, const bf16 *, const bf16 *, const bf16 *, const int *, int, int) = moe_block;
#endif
    blk(y->w1 + e * D * 2 * W, y->b1 + e * 2 * W,
        y->w2 + e * W * D,     y->b2 + e * D,
        R.wlist + R.wchunk[c].off, R.wchunk[c].cnt, (int)e);
}
#else
static void ph_c(void *ctx, size_t c) {
    (void)ctx;
    Layer *y = &M.L[R.layer];
    size_t e = (size_t)R.wchunk[c].e;
    const int W = g_ff;
    const bf16 *w1 = y->w1 + e * D * 2 * W;
    const bf16 *b1 = y->b1 + e * 2 * W;
    const bf16 *w2 = y->w2 + e * W * D;
    const bf16 *b2 = y->b2 + e * D;
    float h1[2 * FF], h[FF];
    double imp[FF];
    if (g_calib) memset(imp, 0, sizeof imp);
    const int *blk = R.wlist + R.wchunk[c].off;
    for (int i2 = 0; i2 < R.wchunk[c].cnt; i2++) {
        int it = blk[i2], t = it / TOPK;
        const float *tb = R.tb + (size_t)t * D;
        cvt_bf16(b1, h1, 2 * W);
        for (int i = 0; i < D; i += 8)
            axpy8_bf16(tb + i, w1 + (size_t)i * 2 * W, 2 * W, h1, 2 * W);
        swiglu_act(h1, h, W);
        float gw = R.gates[it];
        if (g_calib) for (int j = 0; j < W; j++) imp[j] += gw * fabsf(h[j]);
        float *o = R.moe + (size_t)it * D;
        cvt_bf16(b2, o, D);
        for (int i = 0; i < W; i += 8)
            axpy8_bf16(h + i, w2 + (size_t)i * D, D, o, D);
        for (int i = 0; i < D; i++) o[i] *= gw;
    }
    if (g_calib) calib_add((int)e, imp);
}
#endif

/* final phase: moe reduce + final norm + head + log_softmax, per token */
static void ph_f(void *ctx, size_t t) {
    (void)ctx;
    float *x = R.x + t * D;
    const float *m = R.moe + t * TOPK * D;
    for (int s = 0; s < TOPK; s++)
        for (int i = 0; i < D; i++) x[i] += m[s * D + i];
    float tb[D];
    bf16 th[D];
    (void)th;
    rms_norm(x, M.fnorm, tb);
    CVT_H(tb, th, D);
    float *lp = R.lp + t * NC;
    float mx = -1e30f;
    for (int c = 0; c < NC; c++) {
        lp[c] = DOT_X(tb, th, M.unemb + (size_t)c * D, D);
        if (lp[c] > mx) mx = lp[c];
    }
    float den = 0;
    for (int c = 0; c < NC; c++) den += expf(lp[c] - mx);
    float lden = logf(den) + mx;
    for (int c = 0; c < NC; c++) lp[c] -= lden;
}

static void ph_embed(void *ctx, size_t t) {
    (void)ctx;
    cvt_bf16(M.emb + (size_t)R.ids[t] * D, R.x + t * D, D);
}

/* run one window of T tokens; log-probs land in R.lp */
static double ph_ms[4];
/* one packed batch: R.pos/sst/sen must describe the segment layout */
static void forward(const int32_t *ids, int T) {
    R.ids = ids; R.T = T;
    pfor(T, ph_embed, NULL);
    for (int l = 0; l < NL; l++) {
        R.layer = l;
        double u0 = now_ms();
#if defined(HAVE_SME)
        if (g_sme) pfor(((size_t)T + DB - 1) / DB, ph_a_sme, NULL);
        else
#endif
        pfor(T, ph_a, NULL);
        double u1 = now_ms();
#if defined(HAVE_SME)
        if (g_sme) pfor(((size_t)T + DB - 1) / DB, ph_b_sme, NULL);
        else
#endif
        pfor(T, ph_b, NULL);
        double u2 = now_ms();
        /* bucket (token,slot) pairs by expert, then split into <=MOE_B chunks
         * so hot experts don't serialize on one core */
        for (int e = 0; e < NE; e++) R.bucket[e] = -1;
        for (int it = T * TOPK - 1; it >= 0; it--) {
            int e = R.eidx[it];
            if (e < 0) continue;                 /* skipped by --skip-beta */
            R.chain[it] = R.bucket[e];
            R.bucket[e] = it;
        }
        int w = 0, nc = 0;
        for (int e = 0; e < NE; e++) {
            int start = w;
            for (int it = R.bucket[e]; it >= 0; it = R.chain[it]) {
                R.wlist[w++] = it;
                if (w - start == MOE_B) {
                    R.wchunk[nc++] = (struct WChunk){e, start, MOE_B};
                    start = w;
                }
            }
            if (w > start)
                R.wchunk[nc++] = (struct WChunk){e, start, w - start};
        }
        R.nchunk = nc;
        pfor((size_t)nc, g_q8 && l >= Q8_FIRST ? ph_c_q8 : ph_c, NULL);
        double u3 = now_ms();
        ph_ms[0] += u1 - u0; ph_ms[1] += u2 - u1; ph_ms[2] += u3 - u2;
    }
    double u3 = now_ms();
    pfor(T, ph_f, NULL);
    ph_ms[3] += now_ms() - u3;
}

/* ---------------------------------------------------------------- viterbi */
/* label id: 0=O, else 1+cat*4+tag with tag 0..3 = B,I,E,S */
#define TAG(c)  (((c) - 1) & 3)
#define CATOF(c) (((c) - 1) >> 2)
#define IS_O(c) ((c) == 0)

static float vt_trans[NC][NC], vt_start[NC], vt_end[NC];

static void viterbi_init(void) {
    for (int c = 0; c < NC; c++) {
        int tag = IS_O(c) ? -1 : TAG(c);
        vt_start[c] = (IS_O(c) || tag == 0 || tag == 3) ? 0 : NEG_INF; /* O,B,S */
        vt_end[c]   = (IS_O(c) || tag == 2 || tag == 3) ? 0 : NEG_INF; /* O,E,S */
        for (int d = 0; d < NC; d++) {
            int ok, dt = IS_O(d) ? -1 : TAG(d);
            if (IS_O(c) || tag == 2 || tag == 3)          /* O,E,S -> O,B,S   */
                ok = IS_O(d) || dt == 0 || dt == 3;
            else                                          /* B,I -> I,E same  */
                ok = !IS_O(d) && CATOF(c) == CATOF(d) && (dt == 1 || dt == 2);
            vt_trans[c][d] = ok ? 0 : NEG_INF;
            /* six calibration biases would be added here; shipped values are 0 */
        }
    }
}

static void viterbi(const float *lp, int T, uint8_t *out) {
    float s0[NC], s1[NC];
    uint8_t *bp = malloc((size_t)T * NC);
    for (int c = 0; c < NC; c++) s0[c] = lp[c] + vt_start[c];
    for (int t = 1; t < T; t++) {
        for (int d = 0; d < NC; d++) {
            float best = NEG_INF * 2; int barg = 0;
            for (int c = 0; c < NC; c++) {
                float v = s0[c] + vt_trans[c][d];
                if (v > best) { best = v; barg = c; }
            }
            s1[d] = best + lp[t * NC + d];
            bp[t * NC + d] = (uint8_t)barg;
        }
        memcpy(s0, s1, sizeof s0);
    }
    int best = 0; float bv = -1e30f;
    for (int c = 0; c < NC; c++)
        if (s0[c] + vt_end[c] > bv) { bv = s0[c] + vt_end[c]; best = c; }
    out[T - 1] = (uint8_t)best;
    for (int t = T - 1; t > 0; t--) out[t - 1] = bp[t * NC + out[t]];
    free(bp);
}

static void argmax_labels(const float *lp, int T, uint8_t *out) {
    for (int t = 0; t < T; t++) {
        int best = 0;
        for (int c = 1; c < NC; c++)
            if (lp[t * NC + c] > lp[t * NC + best]) best = c;
        out[t] = (uint8_t)best;
    }
}

/* ------------------------------------------------------------------ spans */
typedef struct { int cat; uint32_t bs, be; } Span;

/* mirror of reference labels_to_spans over contiguous labels */
static size_t labels_to_spans(const uint8_t *lab, const TokBuf *tk, Span *sp) {
    size_t ns = 0;
    int cur = -1; uint32_t start = 0; uint32_t prev_end = 0;
#define CLOSE() do { if (cur >= 0) { sp[ns++] = (Span){cur, start, prev_end}; cur = -1; } } while (0)
    for (size_t t = 0; t < tk->n; t++) {
        int c = lab[t];
        if (IS_O(c)) { CLOSE(); prev_end = tk->be[t]; continue; }
        int cat = CATOF(c), tag = TAG(c);
        if (tag == 3) {                       /* S */
            CLOSE();
            sp[ns++] = (Span){cat, tk->bs[t], tk->be[t]};
        } else if (tag == 0) {                /* B */
            CLOSE();
            cur = cat; start = tk->bs[t];
        } else if (tag == 1) {                /* I */
            if (cur != cat) { CLOSE(); cur = cat; start = tk->bs[t]; }
        } else {                              /* E */
            if (cur == cat) {
                sp[ns++] = (Span){cat, start, tk->be[t]};
                cur = -1;
            } else {
                CLOSE();
                sp[ns++] = (Span){cat, tk->bs[t], tk->be[t]};
            }
        }
        prev_end = tk->be[t];
    }
    CLOSE();
#undef CLOSE
    return ns;
}

/* snap byte range outward to UTF-8 char boundaries (reference char-span semantics) */
static void snap_utf8(const uint8_t *s, size_t n, Span *sp) {
    while (sp->bs > 0 && (s[sp->bs] & 0xC0) == 0x80) sp->bs--;
    while (sp->be < n && (s[sp->be] & 0xC0) == 0x80) sp->be++;
}

static int is_trim_ws(uint32_t cp) {
    /* python str.isspace(): White_Space plus 0x1c-0x1f */
    return (uc_flags(cp) & FL_WS) || (cp >= 0x1C && cp <= 0x1F);
}

static void trim_ws(const uint8_t *s, Span *sp) {
    while (sp->bs < sp->be) {
        uint32_t cp; int l = utf8_dec(s + sp->bs, sp->be - sp->bs, &cp);
        if (!is_trim_ws(cp)) break;
        sp->bs += l;
    }
    while (sp->be > sp->bs) {
        uint32_t e = sp->be - 1;
        while (e > sp->bs && (s[e] & 0xC0) == 0x80) e--;
        uint32_t cp; utf8_dec(s + e, sp->be - e, &cp);
        if (!is_trim_ws(cp)) break;
        sp->be = e;
    }
}

static int span_cmp_render(const void *a, const void *b) {
    const Span *x = a, *y = b;
    if (x->bs != y->bs) return x->bs < y->bs ? -1 : 1;
    uint32_t lx = x->be - x->bs, ly = y->be - y->bs;
    if (lx != ly) return lx > ly ? -1 : 1;   /* longer first */
    return strcmp(CAT[x->cat], CAT[y->cat]);
}

/* discard_overlapping_spans_by_label + _select_non_overlapping_spans */
static size_t select_spans(Span *sp, size_t ns) {
    /* per-label overlap discard: sort (start, -len) within label, keep greedily */
    Span *kept = malloc(ns * sizeof *sp);
    size_t nk = 0;
    for (int cat = 0; cat < NCAT; cat++) {
        Span tmp[512]; size_t nt = 0;
        for (size_t i = 0; i < ns && nt < 512; i++)
            if (sp[i].cat == cat) tmp[nt++] = sp[i];
        qsort(tmp, nt, sizeof *tmp, span_cmp_render);
        for (size_t i = 0; i < nt; i++) {
            int ov = 0;
            for (size_t j = nk; j-- > 0; ) {
                if (kept[j].cat != cat) continue;
                if (!(tmp[i].be <= kept[j].bs || tmp[i].bs >= kept[j].be)) { ov = 1; break; }
            }
            if (!ov) kept[nk++] = tmp[i];
        }
    }
    /* global non-overlap selection, left-to-right */
    qsort(kept, nk, sizeof *kept, span_cmp_render);
    size_t no = 0; uint32_t cursor = 0;
    for (size_t i = 0; i < nk; i++) {
        if (kept[i].bs < cursor || kept[i].be <= kept[i].bs) continue;
        sp[no++] = kept[i];
        cursor = kept[i].be;
    }
    free(kept);
    return no;
}

/* ------------------------------------------------------------------- main */
static void print_placeholder(int cat) {
    putchar('<');
    for (const char *p = CAT[cat]; *p; p++) putchar(*p & ~0x20);
    putchar('>');
}

/* labels -> final render-ready spans for one document */
static size_t doc_spans(const char *text, size_t text_len, const TokBuf *tk,
                        const float *lp, int use_argmax, Span *sp) {
    if (tk->n == 0) return 0;
    uint8_t *lab = malloc(tk->n);
    if (use_argmax) argmax_labels(lp, (int)tk->n, lab);
    else            viterbi(lp, (int)tk->n, lab);
    size_t ns = labels_to_spans(lab, tk, sp);
    for (size_t i = 0; i < ns; i++) {
        snap_utf8((const uint8_t *)text, text_len, &sp[i]);
        trim_ws((const uint8_t *)text, &sp[i]);
    }
    size_t w = 0;
    for (size_t i = 0; i < ns; i++) if (sp[i].be > sp[i].bs) sp[w++] = sp[i];
    free(lab);
    return select_spans(sp, w);
}

/* growable byte buffer */
typedef struct { char *p; size_t n, cap; } SB;
static void sb_put(SB *b, const void *s, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->cap ? b->cap * 2 : 4096);
        while (b->cap < b->n + n) b->cap *= 2;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, s, n); b->n += n;
}
static void sb_str(SB *b, const char *s) { sb_put(b, s, strlen(s)); }

/* masked text for one document into a buffer */
static void mask_doc_sb(const char *text, size_t text_len, const TokBuf *tk,
                        const float *lp, int use_argmax, SB *out) {
    Span *sp = malloc((tk->n + 1) * sizeof *sp);
    size_t ns = doc_spans(text, text_len, tk, lp, use_argmax, sp);
    size_t pos = 0;
    for (size_t i = 0; i < ns; i++) {
        sb_put(out, text + pos, sp[i].bs - pos);
        sb_str(out, "<");
        for (const char *p = CAT[sp[i].cat]; *p; p++) { char c = *p & ~0x20; sb_put(out, &c, 1); }
        sb_str(out, ">");
        pos = sp[i].be;
    }
    sb_put(out, text + pos, text_len - pos);
    free(sp);
}

/* decode one document's logprobs and print masked text / JSON */
static void emit_doc(const char *text, size_t text_len, const TokBuf *tk,
                     const float *lp, int json, int use_argmax) {
    if (tk->n == 0 && !json) {
        fwrite(text, 1, text_len, stdout); putchar('\n');
        return;
    }
    Span *sp = malloc((tk->n + 1) * sizeof *sp);
    size_t ns = doc_spans(text, text_len, tk, lp, use_argmax, sp);
    if (json) {
        printf("{\"spans\":[");
        for (size_t i = 0; i < ns; i++) {
            printf("%s{\"label\":\"%s\",\"start\":%u,\"end\":%u,\"text\":\"",
                   i ? "," : "", CAT[sp[i].cat], sp[i].bs, sp[i].be);
            for (uint32_t b = sp[i].bs; b < sp[i].be; b++) {
                uint8_t c = (uint8_t)text[b];
                if (c == '"' || c == '\\') printf("\\%c", c);
                else if (c < 0x20) printf("\\u%04x", c);
                else putchar(c);
            }
            printf("\"}");
        }
        printf("]}\n");
    } else {
        uint32_t pos = 0;
        for (size_t i = 0; i < ns; i++) {
            fwrite(text + pos, 1, sp[i].bs - pos, stdout);
            print_placeholder(sp[i].cat);
            pos = sp[i].be;
        }
        fwrite(text + pos, 1, text_len - pos, stdout);
        putchar('\n');   /* reference CLI always terminates with a newline */
    }
    free(sp);
}

/* ----------------------------------------------------------------- engine */
typedef struct { const char *txt; size_t len; TokBuf tk; float *lp; } Doc;

static int g_cap, g_nbatch;
static int32_t *g_bids;

static void engine_init(int n_ctx, int cap) {
    if (cap < n_ctx) cap = n_ctx;
    g_cap = cap;
    buf_alloc(cap);
    rope_init(n_ctx < cap ? n_ctx : cap);
    g_bids = malloc((size_t)cap * 4);
}

/* tokenized docs in -> per-doc logprobs out, packing segments into batches */
static void run_docs(Doc *docs, size_t ndoc, int n_ctx) {
    size_t total = 0;
    for (size_t d = 0; d < ndoc; d++) {
        docs[d].lp = malloc(docs[d].tk.n * NC * 4 + 4);
        total += docs[d].tk.n;
    }
    struct Seg { size_t doc, off; int len, bat_off; } *segq =
        malloc((total / (size_t)n_ctx + ndoc + 1) * sizeof *segq);
    size_t d = 0, doff = 0;
    while (d < ndoc) {
        int bt = 0, nseg = 0;
        while (d < ndoc) {                       /* fill one packed batch */
            if (doff >= docs[d].tk.n) { d++; doff = 0; continue; }
            int len = (int)(docs[d].tk.n - doff);
            if (len > n_ctx) len = n_ctx;
            if (bt + len > g_cap) break;
            segq[nseg] = (struct Seg){d, doff, len, bt};
            memcpy(g_bids + bt, docs[d].tk.id + doff, (size_t)len * 4);
            for (int i = 0; i < len; i++) {
                R.pos[bt + i] = i; R.sst[bt + i] = bt; R.sen[bt + i] = bt + len;
            }
            bt += len; nseg++; doff += len;
        }
        if (bt == 0) break;
        forward(g_bids, bt);
        g_nbatch++;
        for (int s = 0; s < nseg; s++)
            memcpy(docs[segq[s].doc].lp + segq[s].off * NC,
                   R.lp + (size_t)segq[s].bat_off * NC, (size_t)segq[s].len * NC * 4);
    }
    free(segq);
}

static void doc_free(Doc *d) {
    free(d->tk.id); free(d->tk.bs); free(d->tk.be); free(d->lp);
    memset(&d->tk, 0, sizeof d->tk); d->lp = NULL;
}

/* -------------------------------------------------------- JSON field walker
 * Minimal JSON parser recording every string VALUE whose dotted path matches
 * an adapter pattern ('*' = any one object key or array index). */
typedef struct { size_t vs, ve; char *txt; size_t len; } JField;
typedef struct {
    const char *s; size_t n, i;
    const char *const *pats;
    const char *seg[32]; int seglen[32]; int depth;
    JField *f; size_t nf, fcap;
    int err;
} JW;

static void jw_ws(JW *w) {
    while (w->i < w->n && (w->s[w->i] == ' ' || w->s[w->i] == '\t' ||
                           w->s[w->i] == '\n' || w->s[w->i] == '\r')) w->i++;
}

static int jw_rawstring(JW *w, size_t *vs, size_t *ve) {
    *vs = w->i; w->i++;
    while (w->i < w->n) {
        char c = w->s[w->i];
        if (c == '\\') { w->i += 2; continue; }
        w->i++;
        if (c == '"') { *ve = w->i; return 1; }
    }
    w->err = 1; return 0;
}

static int jw_match(const JW *w) {
    for (int p = 0; w->pats[p]; p++) {
        const char *q = w->pats[p]; int d = 0, ok = 1;
        for (;;) {
            const char *seg = q;
            while (*q && *q != '.') q++;
            int sl = (int)(q - seg);
            if (d >= w->depth || d >= 32) { ok = 0; break; }
            if (!(sl == 1 && seg[0] == '*') &&
                (w->seglen[d] != sl || memcmp(w->seg[d], seg, (size_t)sl) != 0)) { ok = 0; break; }
            d++;
            if (!*q) break;
            q++;
        }
        if (ok && d == w->depth) return 1;
    }
    return 0;
}

/* decode JSON string contents (between quotes) to UTF-8 */
static char *json_decode(const char *s, size_t n, size_t *outlen) {
    char *o = malloc(n + 4); size_t k = 0;
    for (size_t i = 0; i < n; ) {
        if (s[i] != '\\') { o[k++] = s[i++]; continue; }
        if (++i >= n) break;
        char c = s[i++];
        switch (c) {
        case 'b': o[k++] = '\b'; break; case 'f': o[k++] = '\f'; break;
        case 'n': o[k++] = '\n'; break; case 'r': o[k++] = '\r'; break;
        case 't': o[k++] = '\t'; break;
        case 'u': {
            if (i + 4 > n) { i = n; break; }
            uint32_t cp = (uint32_t)strtoul((char[]){s[i],s[i+1],s[i+2],s[i+3],0}, NULL, 16);
            i += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= n && s[i] == '\\' && s[i+1] == 'u') {
                uint32_t lo = (uint32_t)strtoul((char[]){s[i+2],s[i+3],s[i+4],s[i+5],0}, NULL, 16);
                if (lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
            }
            if (cp < 0x80) o[k++] = (char)cp;
            else if (cp < 0x800) { o[k++] = (char)(0xC0|(cp>>6)); o[k++] = (char)(0x80|(cp&0x3F)); }
            else if (cp < 0x10000) { o[k++] = (char)(0xE0|(cp>>12)); o[k++] = (char)(0x80|((cp>>6)&0x3F)); o[k++] = (char)(0x80|(cp&0x3F)); }
            else { o[k++] = (char)(0xF0|(cp>>18)); o[k++] = (char)(0x80|((cp>>12)&0x3F)); o[k++] = (char)(0x80|((cp>>6)&0x3F)); o[k++] = (char)(0x80|(cp&0x3F)); }
            break;
        }
        default: o[k++] = c;    /* covers \" \\ \/ */
        }
    }
    o[k] = 0; *outlen = k;
    return o;
}

static void json_encode_sb(SB *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t c = (uint8_t)s[i];
        if (c == '"') sb_str(b, "\\\"");
        else if (c == '\\') sb_str(b, "\\\\");
        else if (c == '\n') sb_str(b, "\\n");
        else if (c == '\r') sb_str(b, "\\r");
        else if (c == '\t') sb_str(b, "\\t");
        else if (c < 0x20) { char u[8]; snprintf(u, sizeof u, "\\u%04x", c); sb_str(b, u); }
        else sb_put(b, &c, 1);
    }
}

static void jw_value(JW *w) {
    jw_ws(w);
    if (w->err || w->i >= w->n) { w->err = 1; return; }
    char c = w->s[w->i];
    if (c == '"') {
        size_t vs, ve;
        if (!jw_rawstring(w, &vs, &ve)) return;
        if (jw_match(w)) {
            if (w->nf == w->fcap) {
                w->fcap = w->fcap ? w->fcap * 2 : 16;
                w->f = realloc(w->f, w->fcap * sizeof *w->f);
            }
            JField *fl = &w->f[w->nf++];
            fl->vs = vs; fl->ve = ve;
            fl->txt = json_decode(w->s + vs + 1, ve - vs - 2, &fl->len);
        }
    } else if (c == '{') {
        w->i++; jw_ws(w);
        if (w->i < w->n && w->s[w->i] == '}') { w->i++; return; }
        for (;;) {
            jw_ws(w);
            if (w->i >= w->n || w->s[w->i] != '"') { w->err = 1; return; }
            size_t ks, ke;
            if (!jw_rawstring(w, &ks, &ke)) return;
            jw_ws(w);
            if (w->i >= w->n || w->s[w->i] != ':') { w->err = 1; return; }
            w->i++;
            if (w->depth < 32) {
                w->seg[w->depth] = w->s + ks + 1;
                w->seglen[w->depth] = (int)(ke - ks - 2);
            }
            w->depth++;
            jw_value(w);
            w->depth--;
            if (w->err) return;
            jw_ws(w);
            if (w->i < w->n && w->s[w->i] == ',') { w->i++; continue; }
            if (w->i < w->n && w->s[w->i] == '}') { w->i++; return; }
            w->err = 1; return;
        }
    } else if (c == '[') {
        w->i++; jw_ws(w);
        if (w->i < w->n && w->s[w->i] == ']') { w->i++; return; }
        for (;;) {
            if (w->depth < 32) w->seglen[w->depth] = -1;
            w->depth++;
            jw_value(w);
            w->depth--;
            if (w->err) return;
            jw_ws(w);
            if (w->i < w->n && w->s[w->i] == ',') { w->i++; continue; }
            if (w->i < w->n && w->s[w->i] == ']') { w->i++; return; }
            w->err = 1; return;
        }
    } else {   /* number / true / false / null */
        while (w->i < w->n && !strchr(",]} \t\n\r", w->s[w->i])) w->i++;
    }
}

/* --------------------------------------------------------- format adapters
 * An adapter is a route plus the JSON paths whose string values carry user
 * text. The server depends only on this interface; adding a format is one
 * table row. The response is the request payload with those values masked. */
typedef struct { const char *route; const char *const *pats; } Adapter;
static const char *const PAT_PLAIN[]  = {"text", "texts.*", NULL};
static const char *const PAT_OPENAI[] = {"messages.*.content", "messages.*.content.*.text", NULL};
static const char *const PAT_ANTHROPIC[] =
    {"system", "system.*.text", "messages.*.content", "messages.*.content.*.text", NULL};
static const Adapter ADAPTERS[] = {
    {"/redact",           PAT_PLAIN},
    {"/redact/openai",    PAT_OPENAI},
    {"/redact/anthropic", PAT_ANTHROPIC},
};

/* ---------------------------------------------------- cross-request batcher
 * Connection threads tokenize and enqueue jobs; one batcher thread drains the
 * queue after a short linger window and runs a single packed forward over
 * every document in flight. Only the batcher touches the model buffers. */
typedef struct Job { Doc *docs; size_t ndoc; int done; struct Job *next; } Job;
static pthread_mutex_t bq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  bq_new  = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  bq_done = PTHREAD_COND_INITIALIZER;
static Job *bq_head, *bq_tail;
static int g_nctx_srv, g_argmax_srv, g_linger_us = 8000;

static void *batcher_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&bq_mu);
        while (!bq_head) pthread_cond_wait(&bq_new, &bq_mu);
        pthread_mutex_unlock(&bq_mu);
        if (g_linger_us > 0) usleep((useconds_t)g_linger_us);   /* let peers pile in */
        pthread_mutex_lock(&bq_mu);
        Job *jobs = bq_head;
        bq_head = bq_tail = NULL;
        pthread_mutex_unlock(&bq_mu);
        size_t total = 0;
        for (Job *j = jobs; j; j = j->next) total += j->ndoc;
        Doc *all = malloc((total ? total : 1) * sizeof *all);
        size_t k = 0;
        for (Job *j = jobs; j; j = j->next)
            for (size_t d = 0; d < j->ndoc; d++) all[k++] = j->docs[d];
        run_docs(all, total, g_nctx_srv);
        k = 0;
        pthread_mutex_lock(&bq_mu);
        for (Job *j = jobs; j; j = j->next) {
            for (size_t d = 0; d < j->ndoc; d++) j->docs[d].lp = all[k++].lp;
            j->done = 1;
        }
        pthread_cond_broadcast(&bq_done);
        pthread_mutex_unlock(&bq_mu);
        free(all);
    }
    return NULL;
}

static void submit_and_wait(Doc *docs, size_t ndoc) {
    Job job = { docs, ndoc, 0, NULL };
    pthread_mutex_lock(&bq_mu);
    if (bq_tail) bq_tail->next = &job; else bq_head = &job;
    bq_tail = &job;
    pthread_cond_signal(&bq_new);
    while (!job.done) pthread_cond_wait(&bq_done, &bq_mu);
    pthread_mutex_unlock(&bq_mu);
}

/* ------------------------------------------------------------- HTTP server */
static void http_send(int fd, const char *status, const char *body, size_t blen) {
    char hdr[256];
    int hl = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %s\r\nContent-Type: application/json\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", status, blen);
    if (write(fd, hdr, (size_t)hl) < 0) return;
    if (write(fd, body, blen) < 0) return;
}

static void http_handle(int fd, int use_argmax) {
    SB req = {0};
    size_t hdr_end = 0, body_len = 0;
    for (;;) {
        char buf[65536];
        ssize_t r = read(fd, buf, sizeof buf);
        if (r <= 0) { free(req.p); return; }
        sb_put(&req, buf, (size_t)r);
        if (!hdr_end) {
            for (size_t i = 3; i < req.n; i++)
                if (!memcmp(req.p + i - 3, "\r\n\r\n", 4)) { hdr_end = i + 1; break; }
            if (hdr_end) {
                for (size_t i = 0; i + 16 < hdr_end; i++)
                    if (!strncasecmp(req.p + i, "content-length:", 15)) {
                        body_len = strtoull(req.p + i + 15, NULL, 10);
                        break;
                    }
                if (body_len > (64u << 20)) {
                    http_send(fd, "413 Payload Too Large", "{\"error\":\"too large\"}", 21);
                    free(req.p); return;
                }
            }
        }
        if (hdr_end && req.n >= hdr_end + body_len) break;
    }
    char method[8] = {0}, path[128] = {0};
    sscanf(req.p, "%7s %127s", method, path);
    if (!strcmp(method, "GET")) {
        http_send(fd, "200 OK", "{\"status\":\"ok\"}", 15);
        free(req.p); return;
    }
    const Adapter *ad = NULL;
    for (size_t i = 0; i < sizeof ADAPTERS / sizeof *ADAPTERS; i++)
        if (!strcmp(path, ADAPTERS[i].route)) ad = &ADAPTERS[i];
    if (strcmp(method, "POST") != 0 || !ad) {
        http_send(fd, "404 Not Found", "{\"error\":\"unknown route\"}", 25);
        free(req.p); return;
    }
    JW w = { .s = req.p + hdr_end, .n = body_len, .pats = ad->pats,
             .i = 0, .depth = 0, .f = NULL, .nf = 0, .fcap = 0, .err = 0 };
    jw_value(&w);
    if (w.err) {
        http_send(fd, "400 Bad Request", "{\"error\":\"invalid JSON\"}", 24);
        goto out;
    }
    {
        Doc *docs = calloc(w.nf ? w.nf : 1, sizeof *docs);
        for (size_t i = 0; i < w.nf; i++) {
            docs[i].txt = w.f[i].txt; docs[i].len = w.f[i].len;
            tokenize((const uint8_t *)docs[i].txt, docs[i].len, &docs[i].tk);
        }
        submit_and_wait(docs, w.nf);
        SB out2 = {0};
        size_t pos = 0;
        for (size_t i = 0; i < w.nf; i++) {       /* splice masked strings back */
            sb_put(&out2, w.s + pos, w.f[i].vs - pos);
            SB masked = {0};
            mask_doc_sb(docs[i].txt, docs[i].len, &docs[i].tk, docs[i].lp, use_argmax, &masked);
            sb_str(&out2, "\"");
            json_encode_sb(&out2, masked.p, masked.n);
            sb_str(&out2, "\"");
            free(masked.p);
            pos = w.f[i].ve;
        }
        sb_put(&out2, w.s + pos, w.n - pos);
        http_send(fd, "200 OK", out2.p, out2.n);
        free(out2.p);
        for (size_t i = 0; i < w.nf; i++) doc_free(&docs[i]);
        free(docs);
    }
out:
    for (size_t i = 0; i < w.nf; i++) free(w.f[i].txt);
    free(w.f); free(req.p);
}

static void *conn_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    struct timeval tv = {30, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    http_handle(fd, g_argmax_srv);
    close(fd);
    return NULL;
}

static void http_serve(int port, int n_ctx, int use_argmax) {
    signal(SIGPIPE, SIG_IGN);
    g_nctx_srv = n_ctx; g_argmax_srv = use_argmax;
    spawn(batcher_main, NULL, 1);
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port),
                             .sin_addr = { htonl(INADDR_LOOPBACK) } };
    if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 64)) die("bind/listen failed");
    fprintf(stderr, "pf: serving on http://127.0.0.1:%d  (POST /redact, /redact/openai, /redact/anthropic)\n", port);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) continue;
        spawn(conn_main, (void *)(intptr_t)c, 1);
    }
}


int main(int argc, char **argv) {
    const char *model_dir = getenv("PF_MODEL") ? getenv("PF_MODEL") : "model";
    int n_ctx = 4096, json = 0, use_argmax = 0, stats = 0, dump = 0;
    int lines = 0, batch_tok = 16384, serve_port = 0, exact = 0, q8 = -1;
    const char *file = NULL, *trim = NULL, *calib = NULL;
    char *text = NULL; size_t text_len = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--model") && a + 1 < argc) model_dir = argv[++a];
        else if (!strcmp(argv[a], "--ctx") && a + 1 < argc) n_ctx = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-j") || !strcmp(argv[a], "--json")) json = 1;
        else if (!strcmp(argv[a], "--argmax")) use_argmax = 1;
        else if (!strcmp(argv[a], "--stats")) stats = 1;
        else if (!strcmp(argv[a], "--dump-logprobs")) dump = 1;
        else if (!strcmp(argv[a], "--dump-tokens")) dump = 2;
        else if (!strcmp(argv[a], "--lines") || !strcmp(argv[a], "-l")) lines = 1;
        else if (!strcmp(argv[a], "--serve") && a + 1 < argc) serve_port = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--linger") && a + 1 < argc) g_linger_us = atoi(argv[++a]) * 1000;
        else if (!strcmp(argv[a], "--skip-beta") && a + 1 < argc) g_skip_beta = (float)atof(argv[++a]);
        else if (!strcmp(argv[a], "--trim") && a + 1 < argc) trim = argv[++a];
        else if (!strcmp(argv[a], "--calib") && a + 1 < argc) calib = argv[++a];
        else if (!strcmp(argv[a], "--exact")) exact = 1;
        else if (!strcmp(argv[a], "--int8")) q8 = 1;
        else if (!strcmp(argv[a], "--no-int8")) q8 = 0;
        else if (!strcmp(argv[a], "--batch") && a + 1 < argc) batch_tok = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-f") && a + 1 < argc) file = argv[++a];
        else {
            size_t l = strlen(argv[a]);
            text = realloc(text, text_len + l + 2);
            if (text_len) text[text_len++] = ' ';
            memcpy(text + text_len, argv[a], l);
            text_len += l; text[text_len] = 0;
        }
    }
    if (trim && calib) die("--calib needs the full model (drop --trim)");
    /* int8 experts by default where the default is already near-exact (SME2);
     * --exact, --calib and non-SME2 chips keep bf16 unless --int8 forces it */
    if (q8 < 0) q8 = !exact && !calib && sme2_usable();
    if (q8 && calib) die("--calib needs the bf16 experts (drop --int8)");
    g_q8 = q8;
    if (serve_port) {
        if (calib) die("--calib is not supported with --serve");
        tk_load(model_dir);
        model_load(model_dir);
        if (trim) trim_load(trim);
        if (g_q8) q8_pack();
        if (!exact) sme_pack();
        viterbi_init();
        engine_init(n_ctx, batch_tok);
        http_serve(serve_port, n_ctx, use_argmax);
        return 0;
    }
    if (file) {
        FILE *f = strcmp(file, "-") ? fopen(file, "rb") : stdin;
        if (!f) die("cannot open input file");
        size_t cap = 1 << 16; text = malloc(cap); text_len = 0;
        for (size_t r; (r = fread(text + text_len, 1, cap - text_len, f)) > 0; ) {
            text_len += r;
            if (text_len == cap) text = realloc(text, cap *= 2);
        }
        if (f != stdin) fclose(f);
        text = realloc(text, text_len + 1); text[text_len] = 0;
    }
    if (!text) {   /* read stdin */
        size_t cap = 1 << 16; text = malloc(cap); text_len = 0;
        for (size_t r; (r = fread(text + text_len, 1, cap - text_len, stdin)) > 0; ) {
            text_len += r;
            if (text_len == cap) text = realloc(text, cap *= 2);
        }
        text[text_len] = 0;
    }
    if (text_len == 0) return 0;

    /* split into documents: whole input, or one per line with --lines */
    size_t ndoc = 1;
    if (lines) {
        ndoc = 0;
        for (size_t i = 0; i < text_len; i++) if (text[i] == '\n') ndoc++;
        if (text_len && text[text_len - 1] != '\n') ndoc++;
    }
    Doc *docs = calloc(ndoc ? ndoc : 1, sizeof *docs);
    if (lines) {
        size_t d = 0, start = 0;
        for (size_t i = 0; i <= text_len; i++)
            if (i == text_len ? start < i : text[i] == '\n') {
                docs[d].txt = text + start; docs[d].len = i - start;
                d++; start = i + 1;
            }
        ndoc = d;
    } else {
        docs[0].txt = text; docs[0].len = text_len;
    }

    double t0 = now_ms();
    tk_load(model_dir);
    size_t total_tok = 0;
    for (size_t d = 0; d < ndoc; d++) {
        tokenize((const uint8_t *)docs[d].txt, docs[d].len, &docs[d].tk);
        total_tok += docs[d].tk.n;
    }
    if (dump == 2) {
        for (size_t d = 0; d < ndoc; d++)
            for (size_t t = 0; t < docs[d].tk.n; t++) printf("%d\n", docs[d].tk.id[t]);
        return 0;
    }
    model_load(model_dir);
    if (trim) trim_load(trim);
    if (g_q8) q8_pack();
    if (!exact) sme_pack();
    if (calib && !(g_calib = calloc(NL, sizeof *g_calib))) die("out of memory");
    viterbi_init();
    double t1 = now_ms();
    engine_init(n_ctx, batch_tok);
    run_docs(docs, ndoc, n_ctx);
    double t2 = now_ms();
    if (calib) calib_write(calib, total_tok);

    if (dump) {
        for (size_t dd = 0; dd < ndoc; dd++)
            for (size_t t = 0; t < docs[dd].tk.n; t++) {
                printf("%zu %d", t, docs[dd].tk.id[t]);
                for (int c = 0; c < NC; c++) printf(" %.6f", docs[dd].lp[t * NC + c]);
                putchar('\n');
            }
        return 0;
    }
    for (size_t dd = 0; dd < ndoc; dd++)
        emit_doc(docs[dd].txt, docs[dd].len, &docs[dd].tk, docs[dd].lp, json, use_argmax);
    double t3 = now_ms();

    if (stats) {
        fprintf(stderr, "phases (%s%s): norm+qkv %.0fms | attn+router %.0fms | experts %.0fms | head %.0fms\n",
                g_sme ? "sme2" : "vector", g_q8 ? ", int8 experts" : "", ph_ms[0], ph_ms[1], ph_ms[2], ph_ms[3]);
        fprintf(stderr,
                "load+tokenize %.0fms | %zu docs, %zu tok, %d batches | forward %.0fms (%.1f tok/s) | decode %.1fms\n",
                t1 - t0, ndoc, total_tok, g_nbatch, t2 - t1,
                total_tok / ((t2 - t1) / 1e3), t3 - t2);
    }
    return 0;
}
