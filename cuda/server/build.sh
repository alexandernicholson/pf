#!/bin/sh
# Build pf_cuda (GPU forward, fused sm90 MoE from moe.cu, tensor-core attention from attn.cu: attn_fwd_f16 = Variant B with F16 operands, fed fp16 q|k|v by the fused qkv epilogue k_qkv_epi, see attn.h) and pf_cpu (same pf.c without PF_CUDA).
# PF_BATCH_INVARIANT=1 builds the batch-invariant engine instead: gemm_bi.cu replaces cuBLAS for the
# dense projections and attn.cu is compiled with -DATTN_BI (see gpu.cu's header). Default: 0.
# Needs CUDA 12.x (nvcc, cuBLAS) and sm_90a (the MoE kernel uses wgmma); defaults to /workspace/env.sh's toolkit.
set -e
cd "$(dirname "$0")"
[ -n "$CUDA_HOME" ] || . /workspace/env.sh
ARCH=${ARCH:-sm_90a}
BI=${PF_BATCH_INVARIANT:-0}
if [ "$BI" = 1 ]; then BIF="-DPF_BATCH_INVARIANT=1"; ABI="-DATTN_BI"; GBO=gemm_bi.o; else BIF=""; ABI=""; GBO=""; fi
nvcc -O3 -arch=$ARCH -std=c++17 -c moe.cu -o moe.o
nvcc -O3 -arch=$ARCH -std=c++17 -DATTN_PREC=1 $ABI -c attn.cu -o attn.o
nvcc -O3 -arch=$ARCH -std=c++17 $BIF -c gpu.cu -o gpu.o
[ "$BI" = 1 ] && nvcc -O3 -arch=$ARCH -std=c++17 -c gemm_bi.cu -o gemm_bi.o
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -DPF_CUDA -c pf.c -o pf_cuda.o
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cuda pf_cuda.o gpu.o moe.o attn.o $GBO \
    -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcublas -lcudart -lstdc++ -lm
gcc -O3 -std=gnu11 -D_GNU_SOURCE -march=native -pthread -o pf_cpu pf.c -lm
echo "built pf_cuda pf_cpu (PF_BATCH_INVARIANT=$BI)"
