#include <cuda_runtime.h>
#include "attention_kernels.h"
#include "cuda_utils.cuh"

#define TILE_SIZE 32

__global__ void tiled_av_kernel(
    const float *A,
    const float *V,
    float *O,
    int M,
    int N,
    int K
) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load tile of A
        int a_col = tile_idx * TILE_SIZE + threadIdx.x;

        if (row < M && a_col < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int v_row = tile_idx * TILE_SIZE + threadIdx.y;
        int v_col = blockIdx.x * TILE_SIZE + threadIdx.x;

        if (v_row < N && v_col < K) {
            tile_V[threadIdx.y][threadIdx.x] = V[v_row * K + v_col];
        } else {
            tile_V[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_V[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        O[row * K + col] = sum;
    }
}

void launch_tiled_av(
    const float* A,
    const float* V,
    float* O,
    int M,
    int N,
    int K
) {
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        (K + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    tiled_av_kernel<<<grid_dim, block_dim>>>(A, V, O, M, N, K);

    CHECK_LAST_CUDA_ERROR();
}
