#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "attention_kernels.h"

// A @ V
__global__ void naive_av_kernel(const float* A, const float* V,
                                float* O, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    float sum = 0.0f;
    for(int i = 0; i < N; i++) {
        sum += A[row * N + i] * V[i * K + col];
    }

    O[row * K + col] = sum;
}

void launch_naive_av(const float* A, const float* V,
                                        float* O, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
             (M + block.y - 1) / block.y);

    naive_av_kernel<<<grid, block>>>(A, V, O, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}
