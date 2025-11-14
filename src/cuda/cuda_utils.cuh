#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "  %s\n", cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_LAST_CUDA_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "[CUDA KERNEL ERROR] at %s:%d\n", __FILE__, __LINE__); \
            fprintf(stderr, "  %s\n", cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        CHECK_LAST_CUDA_ERROR(); \
        CHECK_CUDA(cudaDeviceSynchronize()); \
    } while(0)

#endif
