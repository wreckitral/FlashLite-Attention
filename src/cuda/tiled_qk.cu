#include "cuda_utils.cuh"
#include "attention_kernels.h"
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void tiled_qk_kernel(
    const float* Q,
    const float* K,
    float* S,
    int M,
    int N,
    int d_k,
    float scale
) {
    // Shared memory tiles
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];  // Will store K transposed

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (d_k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load tile of Q (same as before)
        int q_col = tile_idx * TILE_SIZE + threadIdx.x;

        if (row < M && q_col < d_k) {
            tile_Q[threadIdx.y][threadIdx.x] = Q[row * d_k + q_col];
        } else {
            tile_Q[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int k_row = blockIdx.x * TILE_SIZE + threadIdx.y;
        int k_col = tile_idx * TILE_SIZE + threadIdx.x;

        if (k_row < N && k_col < d_k) {
            tile_K[threadIdx.x][threadIdx.y] = K[k_row * d_k + k_col];
        } else {
            tile_K[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_Q[threadIdx.y][k] * tile_K[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        S[row * N + col] = sum * scale;
    }
}

void launch_tiled_qk(
    const float* Q,
    const float* K,
    float* S,
    int M,
    int N,
    int d_k,
    float scale
) {
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    tiled_qk_kernel<<<grid_dim, block_dim>>>(Q, K, S, M, N, d_k, scale);
    CHECK_LAST_CUDA_ERROR();
}
