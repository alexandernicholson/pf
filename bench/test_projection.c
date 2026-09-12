/* Exact projection equivalence, including row/input tails, bias and residual. */
#define main pf_application_main
#include "../pf.c"
#undef main
#include <inttypes.h>

#if defined(HAVE_PROJ4)
static uint32_t prng = 1234567;
static float sample(void) {
    prng ^= prng << 13; prng ^= prng >> 17; prng ^= prng << 5;
    return ((int)(prng % 20001) - 10000) / 1024.0f;
}
static bf16 half(float x) { uint32_t b; memcpy(&b, &x, 4); return b >> 16; }
static float oracle(const float *x, const bf16 *w, int n) {
    float a[8] = {0}, b[8] = {0}; int k = 0;
    for (; k + 16 <= n; k += 16)
        for (int j = 0; j < 8; j++) {
            a[j] = fmaf(x[k+j], b2f(w[k+j]), a[j]);
            b[j] = fmaf(x[k+8+j], b2f(w[k+8+j]), b[j]);
        }
    for (int j = 0; j < 8; j++) a[j] += b[j];
    for (int j = 0; j < 4; j++) a[j] += a[j+4];
    float sum = (a[0] + a[1]) + (a[2] + a[3]);
    for (; k < n; k++) sum = fmaf(x[k], b2f(w[k]), sum);
    return sum;
}
#endif
int main(void) {
#if defined(HAVE_PROJ4)
    int cols[] = {1, 15, 16, 17, 40, 640, 896};
    int rows[] = {1, 3, 4, 7, 8, 9, 33, 128};
    size_t checked = 0, failures = 0;
    for (int c = 0; c < 7; c++) for (int r = 0; r < 8; r++) {
        int N=cols[c], M=rows[r];
        float *x=malloc(N*4), *y=malloc((M+2)*4), *old=malloc(M*4);
        bf16 *w=malloc((size_t)M*N*2), *bias=malloc(M*2);
        if (!x || !y || !old || !w || !bias) return 2;
        for (int i=0;i<N;i++) x[i]=sample();
        for (int i=0;i<M*N;i++) w[i]=half(sample());
        for (int i=0;i<M;i++) {old[i]=sample();bias[i]=half(sample());}
        for (int add=0;add<2;add++) for (int biased=0;biased<2;biased++) {
            y[0]=123.0f; y[M+1]=456.0f; memcpy(y+1,old,M*4);
            project_x86(x,w,biased?bias:NULL,y+1,M,N,add);
            for (int i=0;i<M;i++) {
                float expected=oracle(x,w+(size_t)i*N,N);
                if (biased) expected+=b2f(bias[i]);
                if (add) expected=old[i]+expected;
                failures+=memcmp(y+i+1,&expected,4)!=0; checked++;
            }
            failures+=y[0]!=123.0f || y[M+1]!=456.0f;
        }
        free(x);free(y);free(old);free(w);free(bias);
    }
    printf("{\"checked_outputs\":%zu,\"bitwise_mismatches\":%zu,\"projection_rows\":%d}\n",checked,failures,PF_PROJ_ROWS);
    return failures ? 1 : 0;
#else
    puts("{\"status\":\"skip\",\"reason\":\"projection kernel unavailable\"}");return 77;
#endif
}
