#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include "attention_kernels.h"

__global__ void naive_softmax_kernel(const float* input, float* output, int M,
                                     int N, bool use_causal_mask) {
    __shared__ float shared_data[256];

    int row = blockIdx.x; // one block computing one row
    int tid = threadIdx.x; // thread id

    if(row >= M) return;

    // 1. FIND MAX
    float local_max = -INFINITY;
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        float value = input[idx];

        if (use_causal_mask && col > row) {
            value = -INFINITY;
        }

        local_max = fmaxf(local_max, input[idx]);
    }

    // store max in shared memory with threadId as the index
    shared_data[tid] = local_max;
    __syncthreads();

    // reduction to get max for the entire row
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if(tid < i) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + i]);
        }

        __syncthreads();
    }

    float row_max = shared_data[0];
    __syncthreads();

    // 2. COMPUTE SUM OF ROWS
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        float value = input[idx];

        // apply causal mask
        if (use_causal_mask && col > row) {
            value = -INFINITY;
        }

        local_sum += expf(value - row_max);
    }

    shared_data[tid] = local_sum;
    __syncthreads();

    // reduction to get sum for the entire row
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if(tid < i) {
            shared_data[tid] += shared_data[tid + i];
        }

        __syncthreads();
    }

    float row_sum = shared_data[0];
    __syncthreads();

    // 3. NORMALIZE
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;

        if (use_causal_mask && col > row) {
            output[idx] = 0.0f;
        } else {
            output[idx] = expf(input[idx] - row_max) / row_sum;
        }
    }
}

void launch_naive_softmax(const float* input, float* output,
                          int M, int N,
                          bool use_causal_mask) {
    int threads = 256;
    int blocks = M;

    naive_softmax_kernel<<<blocks, threads>>>(
        input, output, M, N, use_causal_mask
    );
    CHECK_LAST_CUDA_ERROR();
}
