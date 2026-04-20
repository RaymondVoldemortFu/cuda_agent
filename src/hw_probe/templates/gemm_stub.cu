// Minimal CUDA executable for environment / NCU smoke tests (FP32 GEMM-ish kernel).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr;
    if (cudaMalloc(&d_x, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc x failed\n");
        return 1;
    }
    if (cudaMalloc(&d_y, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc y failed\n");
        return 1;
    }
    if (cudaMemset(d_x, 0, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemset x failed\n");
        return 1;
    }
    if (cudaMemset(d_y, 0, bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemset y failed\n");
        return 1;
    }
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(n, 1.0f, d_x, d_y);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel sync failed\n");
        return 1;
    }
    float host = 0.0f;
    if (cudaMemcpy(&host, d_y, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed\n");
        return 1;
    }
    std::printf("ok sum0=%f\n", static_cast<double>(host));
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
