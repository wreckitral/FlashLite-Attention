/*
 * Attention CUDA Kernels Header
 */

#ifndef ATTENTION_KERNELS_H
#define ATTENTION_KERNELS_H

#include <cuda_runtime.h>

// kernel function (GPU)
__global__ void naive_qk_kernel(const float* A, const float* B, float* S,
                                    int M, int N, int K, float scale);

__global__ void naive_softmax_kernel(const float* input, float* output, int M,
                                     int N, bool use_causal_mask);

__global__ void naive_av_kernel(const float* A, const float* V,
                                        float* O, int M, int N, int K);

__global__ void tiled_qk_kernel(const float* Q, const float* K, float* S,
                                int M, int N, int d_k, float scale);

__global__ void tiled_av_kernel(const float* A, const float* V,
                                        float* O, int M, int N, int K);

__global__ void fused_qk_softmax_small_kernel(
    const float* Q,
    const float* K,
    float* A,
    int M,
    int N,
    int d_k,
    float scale
);

__global__ void online_softmax_kernel(const float *Q, const float *K,
                                      float* A, int M, int N, int d_k, float scale,
                                      bool use_causal_mask);

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
);

// host wrapper function (CPU)
void launch_naive_qk(const float* A, const float* B, float* S,
                                    int M, int N, int K, float scale);

void launch_naive_softmax(const float* input, float* output, int M,
                                     int N, bool use_causal_mask);

void launch_naive_av(const float* A, const float* V,
                                        float* O, int M, int N, int K);

void launch_tiled_qk(const float* Q, const float* K, float* S, int M, int N,
                        int d_k, float scale);

void launch_tiled_av(const float* A, const float* V,
                                        float* O, int M, int N, int K);

void launch_fused_qk_softmax_small(
    const float* Q,
    const float* K,
    float* A,
    int M,
    int N,
    int d_k,
    float scale
);

void launch_online_softmax(const float *Q, const float *K,
                                      float* A, int M, int N, int d_k, float scale,
                        bool use_causal_mask);

void launch_flashLite_attention(
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
);

#endif
