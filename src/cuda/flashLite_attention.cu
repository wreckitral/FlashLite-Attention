#include <cuda_runtime.h>
#include <float.h>
#include "cuda_utils.cuh"
#include "attention_kernels.h"

#define TILE_M 16  // Process 16 output rows per block
#define TILE_N 32  // Process 32 K/V columns per iteration
#define D_K 64     // Embedding dimension

__global__ void flashLite_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int M,
    int N,
    int d_k,
    float scale,
    bool use_causal_mask,
    int q_offset
) {
    // Each block processes TILE_M rows of output
    int row_start = blockIdx.x * TILE_M;
    int local_row = threadIdx.y;  // Each warp handles one row
    int global_row = row_start + local_row;

    // Shared memory for tiles
    __shared__ float tile_Q[TILE_M][D_K];
    __shared__ float tile_K[TILE_N][D_K];
    __shared__ float tile_V[TILE_N][D_K];
    __shared__ float tile_S[TILE_M][TILE_N];  // Scores

    // Per-row accumulators in shared memory
    __shared__ float s_max[TILE_M];
    __shared__ float s_sum[TILE_M];
    __shared__ float s_O[TILE_M][D_K];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // Initialize per-row statistics
    if (threadIdx.x == 0 && local_row < TILE_M) {
        s_max[local_row] = -FLT_MAX;
        s_sum[local_row] = 0.0f;
    }

    // Initialize output accumulator
    for (int k = threadIdx.x; k < d_k; k += blockDim.x) {
        if (local_row < TILE_M) {
            s_O[local_row][k] = 0.0f;
        }
    }
    __syncthreads();

    // Load Q tile cooperatively
    for (int idx = tid; idx < TILE_M * d_k; idx += num_threads) {
        int i = idx / d_k;
        int j = idx % d_k;
        int row = row_start + i;
        tile_Q[i][j] = (row < M) ? Q[row * d_k + j] : 0.0f;
    }
    __syncthreads();

    // Iterate through K/V tiles
    int num_tiles = (N + TILE_N - 1) / TILE_N;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int col_start = tile_idx * TILE_N;

        if (use_causal_mask && col_start > row_start + TILE_M - 1) {
            break;
        }

        // Load K and V tiles cooperatively
        for (int idx = tid; idx < TILE_N * d_k; idx += num_threads) {
            int i = idx / d_k;
            int j = idx % d_k;
            int col = col_start + i;

            tile_K[i][j] = (col < N) ? K[col * d_k + j] : 0.0f;
            tile_V[i][j] = (col < N) ? V[col * d_k + j] : 0.0f;
        }
        __syncthreads();

        // Compute scores: tile_S = tile_Q @ tile_K^T
        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int k = 0; k < D_K; k++) {
                    sum += tile_Q[i][k] * tile_K[j][k];
                }
                tile_S[i][j] = sum * scale;
            }
        }
        __syncthreads();

        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
            int global_r = row_start + i;
            if (global_r < M) {
                int query_pos = q_offset + global_r;
                for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                    int global_col = col_start + j;

                    bool is_future = (global_col > query_pos);

                    if (global_col >= N || (use_causal_mask && is_future)) {
                        tile_S[i][j] = -FLT_MAX;
                    }
                }
            }
        }
        __syncthreads();

        // Each row processes its own softmax
        if (global_row < M && local_row < TILE_M) {
            // Find row max using warp shuffle reduction
            float row_max = -FLT_MAX;

            if (global_row < M) {
                for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                    int global_col = col_start + j;

                    bool is_visible = !use_causal_mask || (global_col <= q_offset + global_row);

                    if (global_col < N && is_visible) {
                        row_max = fmaxf(row_max, tile_S[local_row][j]);
                    }
                }
            }

            // Warp-level reduction for max
            for (int offset = 16; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
            }

            // First thread in each row updates (only if row is valid)
            if (threadIdx.x == 0 && global_row < M) {
                float m_old = s_max[local_row];
                float m_new = fmaxf(m_old, row_max);

                // Rescale if max changed
                if (m_new > m_old) {
                    float scale = expf(m_old - m_new);
                    s_sum[local_row] *= scale;

                    // Rescale output
                    #pragma unroll 8
                    for (int k = 0; k < d_k; k++) {
                        s_O[local_row][k] *= scale;
                    }
                }

                s_max[local_row] = m_new;
            }
            __syncthreads();

            // Compute exp and sum
            float m_new = s_max[local_row];
            float thread_sum = 0.0f;

            if (global_row < M) {
                for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) {
                    int global_col = col_start + j;

                    bool is_visible = !use_causal_mask || (global_col <= q_offset + global_row);

                    if (global_col < N && is_visible) {
                        float exp_val = expf(tile_S[local_row][j] - m_new);
                        tile_S[local_row][j] = exp_val;
                        thread_sum += exp_val;
                    } else {
                        // Set masked positions to exactly 0
                        tile_S[local_row][j] = 0.0f;
                    }
                }
            }

            // Warp-level reduction for sum
            for (int offset = 16; offset > 0; offset /= 2) {
                thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
            }

            if (threadIdx.x == 0 && global_row < M) {
                s_sum[local_row] += thread_sum;
            }
            __syncthreads();

            // Accumulate output: O += exp(S) @ V
            if (global_row < M) {
                for (int k = threadIdx.x; k < d_k; k += blockDim.x) {
                    float acc = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < TILE_N; j++) {
                        // Masked positions are already 0, so just multiply
                        acc += tile_S[local_row][j] * tile_V[j][k];
                    }
                    s_O[local_row][k] += acc;
                }
            }
            __syncthreads();
        }
    }

    // Final normalization and write output
    if (global_row < M && local_row < TILE_M) {
        float inv_sum = 1.0f / s_sum[local_row];
        for (int k = threadIdx.x; k < d_k; k += blockDim.x) {
            O[global_row * d_k + k] = s_O[local_row][k] * inv_sum;
        }
    }
}
