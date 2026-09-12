/* Exactly-once and publication checks across consecutive pool epochs. */
#define main pf_cli_main
#include "../pf.c"
#undef main
#include <stdatomic.h>

typedef struct {
    _Atomic unsigned hits[257];
    unsigned values[257];
    unsigned epoch;
} Check;

static void visit(void *ctx, size_t i) {
    Check *c = ctx;
    atomic_fetch_add_explicit(&c->hits[i], 1, memory_order_relaxed);
    c->values[i] = c->epoch * 257 + (unsigned)i;
}

int main(void) {
    const size_t sizes[] = {0, 1, 2, 3, 8, 9, 65, 257};
    Check c = {0};
    for (unsigned epoch = 1; epoch <= 128; epoch++) {
        c.epoch = epoch;
        size_t n = sizes[epoch % (sizeof sizes / sizeof sizes[0])];
        for (size_t i = 0; i < 257; i++) {
            atomic_store(&c.hits[i], 0);
            c.values[i] = 0;
        }
        pfor(n, visit, &c);
        for (size_t i = 0; i < 257; i++) {
            if (atomic_load(&c.hits[i]) != (unsigned)(i < n)) return 1;
            if (c.values[i] != (i < n ? epoch * 257 + i : 0)) return 2;
        }
    }
    puts("{\"pool_epochs\":128,\"exactly_once\":true,\"publication\":true}");
    return 0;
}
