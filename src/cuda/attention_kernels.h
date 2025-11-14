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

// host wrapper function (CPU)
void launch_naive_qk(const float* A, const float* B, float* S,
                                    int M, int N, int K, float scale);

void launch_naive_softmax(const float* input, float* output, int M,
                                     int N, bool use_causal_mask);

void launch_naive_av(const float* A, const float* V,
                                        float* O, int M, int N, int K);


#endif
