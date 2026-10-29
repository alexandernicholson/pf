/* Pipeline index math of k_ws (moe_ws.inc); plain header so tests/test_ws.cpp can use it on the host. */
#pragma once
struct WsSched {
    int ST, KT;
    __host__ __device__ int stage(int g) const { return g % ST; }
    __host__ __device__ unsigned full_parity(int g) const { return (unsigned)(g / ST) & 1u; }
    __host__ __device__ unsigned empty_parity(int g) const { return ((unsigned)(g / ST) & 1u) ^ 1u; }
    /* consumer c's j-th mainloop is local item 2j + c; it waits turn[c] for phase (c ? j : j - 1) */
    __host__ __device__ static bool turn_waits(int c, int j) { return c == 1 || j > 0; }
    __host__ __device__ static unsigned turn_parity(int c, int j) { return (unsigned)(c ? j : j - 1) & 1u; }
};

