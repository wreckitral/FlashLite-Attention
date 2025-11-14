#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "attention_kernels.h"

// Q @ K^T
__global__ void naive_qk_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for(int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[col * K + i];
    }
    C[row * N + col] = sum * scale;
}

void launch_naive_qk(const float* A, const float* B, float* C,
                         int M, int N, int K, float scale) {
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid((N + block.x - 1) / block.x,   // Ceiling division
             (M + block.y - 1) / block.y);

    naive_qk_kernel<<<grid, block>>>(A, B, C, M, N, K, scale);
    CHECK_LAST_CUDA_ERROR();
}
