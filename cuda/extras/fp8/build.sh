#!/bin/sh
# Build pf_cuda (GPU forward, fused sm90 MoE from moe.cu (+ opt-in e4m3 experts, PF_FP8_EXPERTS=1, moe_fp8_sm90.inc), tensor-core attention from attn.cu with ATTN_PREC=1 = F16 operands, see attn.h) and pf_cpu (same pf.c without PF_CUDA).
# Needs CUDA 12.x (nvcc, cuBLAS) and sm_90a (the MoE kernel uses wgmma); defaults to /workspace/env.sh's toolkit.
set -e
cd "$(dirname "$0")"
[ -n "$CUDA_HOME" ] || . /workspace/env.sh
ARCH=${ARCH:-sm_90a}
nvcc -O3 -arch=$ARCH -std=c++17 -c moe.cu -o moe.o
nvcc -O3 -arch=$ARCH -std=c++17 -DATTN_PREC=1 -c attn.cu -o attn.o
nvcc -O3 -arch=$ARCH -std=c++17 -c gpu.cu -o gpu.o
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -DPF_CUDA -c pf.c -o pf_cuda.o
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cuda pf_cuda.o gpu.o moe.o attn.o \
    -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcublas -lcudart -lstdc++ -lm
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cpu pf.c -lm
echo "built pf_cuda pf_cpu"
