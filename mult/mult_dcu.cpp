#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// 矩阵大小
#define N 1024
#define M 2024
#define P 512

// 每个线程块的尺寸
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

// GPU 核函数：每个线程计算一个 C[i][j]
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    }
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > 1e-6) {
            std::cerr << "Mismatch at " << i << ": ref=" << ref[i] << " vs test=" << test[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    // 1. CPU 基准计算并计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_ref);
    auto cpu_end   = std::chrono::high_resolution_clock::now();
    double cpu_ms  = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "[CPU] Time: " << cpu_ms << " ms\n";

    // 2. 设备端指针与内存分配
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = N * M * sizeof(double);
    size_t size_B = M * P * sizeof(double);
    size_t size_C = N * P * sizeof(double);

    // GPU 总体计时开始
    auto gpu_start = std::chrono::high_resolution_clock::now();

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    // 3. 拷贝输入到设备
    hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice);

    // 4. 配置执行参数
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // 5. GPU 核函数计时事件
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    hipLaunchKernelGGL(matmul_kernel,
                       gridDim, blockDim, 0, 0,
                       d_A, d_B, d_C, N, M, P);

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float kernel_ms = 0;
    hipEventElapsedTime(&kernel_ms, start, stop);

    // 6. 拷贝结果回主机
    hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost);

    // GPU 总体计时结束
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "[HIP] Kernel time: " << kernel_ms << " ms\n";
    std::cout << "[HIP] Total GPU time (H2D+kernel+D2H): " << gpu_total_ms << " ms\n";

    // 7. 验证
    bool ok = validate(C_ref, C);
    std::cout << "[HIP] Validation: " << (ok ? "PASS" : "FAIL") << std::endl;

    // 8. 清理
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return ok ? 0 : 1;
}
