#!/bin/sh
# Build pf_cuda (GPU forward, fused sm90 MoE from moe.cu with the warp-specialized persistent GEMMs of moe_ws.inc, config in moe_cfg.h, ptxas report in moe_ptxas.log; tensor-core attention from attn.cu: attn_fwd_f16 = Variant B with F16 operands, fed fp16 q|k|v by the fused qkv epilogue k_qkv_epi, see attn.h; double-buffered forward: gpu_submit/gpu_wait) and pf_cpu (same pf.c without PF_CUDA).
# Needs CUDA 12.x (nvcc, cuBLAS) and sm_90a (the MoE kernel uses wgmma and TMA); defaults to /workspace/env.sh's toolkit.
set -e
cd "$(dirname "$0")"
[ -n "$CUDA_HOME" ] || . /workspace/env.sh
ARCH=${ARCH:-sm_90a}
nvcc -O3 -arch=$ARCH -std=c++17 -Xptxas -v -c moe.cu -o moe.o 2> moe_ptxas.log  # moe_cfg.h picks the GEMM config
nvcc -O3 -arch=$ARCH -std=c++17 -DATTN_PREC=1 -c attn.cu -o attn.o
nvcc -O3 -arch=$ARCH -std=c++17 -c gpu.cu -o gpu.o
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -DPF_CUDA -c pf.c -o pf_cuda.o
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cuda pf_cuda.o gpu.o moe.o attn.o \
    -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcublas -lcudart -lstdc++ -lm
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cpu pf.c -lm
echo "built pf_cuda pf_cpu"
