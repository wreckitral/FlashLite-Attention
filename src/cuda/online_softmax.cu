#include <cuda_runtime.h>
#include <float.h>
#include "cuda_utils.cuh"
#include "attention_kernels.h"

#define TILE_M 32
#define TILE_N 32
#define D_K 64

__global__ void online_softmax_kernel(
    const float* Q,
    const float* K,
    float* A,
    int M,
    int N,
    int d_k,
    float scale,
    bool use_causal_mask
) {
    // Each block processes TILE_M rows of output
    int row_start = blockIdx.x * TILE_M;

    // Shared memory for tiles
    __shared__ float tile_Q[TILE_M][D_K];
    __shared__ float tile_K[TILE_N][D_K];
    __shared__ float tile_S[TILE_M][TILE_N];  // Scores

    // Shared memory for running statistics (per row in this block)
    __shared__ float s_max[TILE_M];     // Running max per row
    __shared__ float s_sum[TILE_M];     // Running sum per row

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // Initialize statistics for all rows in this block
    for (int i = tid; i < TILE_M; i += num_threads) {
        s_max[i] = -FLT_MAX;
        s_sum[i] = 0.0f;
    }
    __syncthreads();

    // Load Q tile (once, reused for all K tiles)
    for (int idx = tid; idx < TILE_M * d_k; idx += num_threads) {
        int i = idx / d_k;
        int j = idx % d_k;
        int global_row = row_start + i;
        if (global_row < M) {
            tile_Q[i][j] = Q[global_row * d_k + j];
        } else {
            tile_Q[i][j] = 0.0f;
        }
    }
    __syncthreads();

    // Number of K tiles to process
    int num_k_tiles = (N + TILE_N - 1) / TILE_N;

    // Iterate through K tiles
    for (int tile_idx = 0; tile_idx < num_k_tiles; tile_idx++) {
        int col_start = tile_idx * TILE_N;

        if (use_causal_mask && col_start > row_start + TILE_M - 1) {
            break;
        }

        // Load K tile
        for (int idx = tid; idx < TILE_N * d_k; idx += num_threads) {
            int i = idx / d_k;
            int j = idx % d_k;
            int global_col = col_start + i;
            if (global_col < N) {
                tile_K[i][j] = K[global_col * d_k + j];
            } else {
                tile_K[i][j] = 0.0f;
            }
        }
        __syncthreads();

        // Compute scores: S = Q @ K^T
        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                float sum = 0.0f;
                for (int k = 0; k < d_k; k++) {
                    sum += tile_Q[i][k] * tile_K[j][k];
                }
                tile_S[i][j] = sum * scale;
            }
        }
        __syncthreads();

        // Apply causal mask: set S[i,j] = -inf if row < col
        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                int global_row = row_start + i;
                int global_col = col_start + j;

                bool is_masked = use_causal_mask && (global_row < global_col);

                if (global_row < M && global_col < N && is_masked) {
                    tile_S[i][j] = -FLT_MAX;
                }
            }
        }
        __syncthreads();

        // Process each row independently for online softmax
        for (int local_row = 0; local_row < TILE_M; local_row++) {
            int global_row = row_start + local_row;
            if (global_row >= M) continue;

            // Find max in this tile for this row
            __shared__ float row_max_shared;
            float thread_max = -FLT_MAX;

            for (int j = tid; j < TILE_N; j += num_threads) {
                int global_col = col_start + j;
                if (global_col < N) {
                    thread_max = fmaxf(thread_max, tile_S[local_row][j]);
                }
            }

            // Reduce to find max across all threads
            __shared__ float reduce_buffer[256];
            reduce_buffer[tid] = thread_max;
            __syncthreads();

            for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_buffer[tid] = fmaxf(reduce_buffer[tid], reduce_buffer[tid + stride]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                row_max_shared = reduce_buffer[0];
            }
            __syncthreads();

            float m_tile = row_max_shared;
            float m_old = s_max[local_row];
            float m_new = fmaxf(m_old, m_tile);

            // If max changed, rescale ALL previous values in global memory
            if (m_new > m_old && tile_idx > 0) {
                float correction = expf(m_old - m_new);

                // Rescale all previously written values
                for (int prev_tile = 0; prev_tile < tile_idx; prev_tile++) {
                    int prev_col_start = prev_tile * TILE_N;

                    for (int j = tid; j < TILE_N; j += num_threads) {
                        int global_col = prev_col_start + j;

                        bool is_valid = !use_causal_mask || (global_row >= global_col);

                        if (global_col < N && is_valid) {
                            A[global_row * N + global_col] *= correction;
                        }
                    }
                }
                __syncthreads();

                // Rescale the sum
                if (tid == 0) {
                    s_sum[local_row] *= correction;
                }
            }

            // Update max
            if (tid == 0) {
                s_max[local_row] = m_new;
            }
            __syncthreads();

            // Compute exp(S - m_new) and sum
            __shared__ float row_sum_shared;
            float thread_sum = 0.0f;

            for (int j = tid; j < TILE_N; j += num_threads) {
                int global_col = col_start + j;
                if (global_col < N) {
                    float exp_val = expf(tile_S[local_row][j] - m_new);
                    tile_S[local_row][j] = exp_val;
                    thread_sum += exp_val;
                }
            }

            // Reduce to find sum
            reduce_buffer[tid] = thread_sum;
            __syncthreads();

            for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    reduce_buffer[tid] += reduce_buffer[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                row_sum_shared = reduce_buffer[0];
            }
            __syncthreads();

            // Update running sum
            if (tid == 0) {
                s_sum[local_row] += row_sum_shared;
            }
            __syncthreads();

            // Write unnormalized exp values to global memory
            for (int j = tid; j < TILE_N; j += num_threads) {
                int global_col = col_start + j;
                if (global_col < N) {
                    A[global_row * N + global_col] = tile_S[local_row][j];
                }
            }
            __syncthreads();
        }
    }

    // Final normalization: divide all values by their row's sum
    for (int tile_idx = 0; tile_idx * TILE_N < N; tile_idx++) {
        int col_start = tile_idx * TILE_N;

        // Skip future tiles logic needs update
        if (use_causal_mask && col_start > row_start + TILE_M - 1) {
            break;
        }

        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            int global_row = row_start + i;
            if (global_row >= M) continue;

            float row_sum = s_sum[i];

            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                int global_col = col_start + j;

                bool is_masked = use_causal_mask && (global_row < global_col);

                if (global_col < N && !is_masked) {
                    // Normalize
                    A[global_row * N + global_col] /= row_sum;
                } else if (global_col < N && is_masked) {
                    // Ensure masked positions are exactly 0
                    A[global_row * N + global_col] = 0.0f;
                }
            }
        }
        __syncthreads();
    }
}

void launch_online_softmax(
    const float* Q,
    const float* K,
    float* A,
    int M,
    int N,
    int d_k,
    float scale,
    bool use_causal_mask
) {
    dim3 block(8, 8);  // 64 threads per block
    dim3 grid((M + TILE_M - 1) / TILE_M);

    online_softmax_kernel<<<grid, block>>>(Q, K, A, M, N, d_k, scale, use_causal_mask);
    CHECK_LAST_CUDA_ERROR();
}
