#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 编译：
//   hipcc sourcefile_mlp_forward.cpp -o mlp_forward
// 执行：
//   ./mlp_forward      或者 hipprof ./mlp_forward

#define BATCH 1024
#define I      10
#define H      20
#define O       5

// 矩阵乘法核函数：C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel(const double* A,
                              const double* B,
                              double*       C,
                              int           M,
                              int           N,
                              int           K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 添加偏置核函数
__global__ void add_bias_kernel(double*         mat,
                                const double*   bias,
                                int             rows,
                                int             cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        mat[idx] += bias[col];
    }
}

// ReLU 激活核函数
__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
}

// 随机初始化
void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

int main() {
    // 主机端数据
    std::vector<double> h_X(BATCH * I),
                        h_W1(I     * H),
                        h_B1(H),
                        h_H(BATCH * H),
                        h_W2(H     * O),
                        h_B2(O),
                        h_Y(BATCH * O);

    // 随机初始化
    srand(42);
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // 设备端指针
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    hipMalloc(&d_X,  BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I     * H * sizeof(double));
    hipMalloc(&d_B1, H     * sizeof(double));
    hipMalloc(&d_H,  BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H     * O * sizeof(double));
    hipMalloc(&d_B2, O     * sizeof(double));
    hipMalloc(&d_Y,  BATCH * O * sizeof(double));

    // CPU 计时起点：从第一次内存拷贝开始
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // 主机 -> 设备
    hipMemcpy(d_X,  h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I     * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H     * sizeof(double),       hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H     * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O     * sizeof(double),       hipMemcpyHostToDevice);

    // GPU 计时事件
    hipEvent_t gpu_start, gpu_stop;
    hipEventCreate(&gpu_start);
    hipEventCreate(&gpu_stop);

    // 在第一个 kernel 前记录 GPU 开始时间
    hipEventRecord(gpu_start, 0);

    // 启动 kernels
    dim3 blockSize(16, 16), gridSize;
    int threadsPerBlock = 256;

    // 隐藏层： H = X * W1
    gridSize.x = (O + blockSize.x - 1) / blockSize.x;  // 这里临时用 O/H 均可
    gridSize.y = (BATCH + blockSize.y - 1) / blockSize.y;
    hipLaunchKernelGGL(matmul_kernel, gridSize, blockSize, 0, 0,
                       d_X, d_W1, d_H, BATCH, H, I);

    // 添加偏置 B1
    int blocksPerGrid = (BATCH * H + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(add_bias_kernel, blocksPerGrid, threadsPerBlock, 0, 0,
                       d_H, d_B1, BATCH, H);

    // ReLU 激活
    hipLaunchKernelGGL(relu_kernel, blocksPerGrid, threadsPerBlock, 0, 0,
                       d_H, BATCH * H);

    // 输出层： Y = H * W2
    gridSize.x = (O + blockSize.x - 1) / blockSize.x;
    gridSize.y = (BATCH + blockSize.y - 1) / blockSize.y;
    hipLaunchKernelGGL(matmul_kernel, gridSize, blockSize, 0, 0,
                       d_H, d_W2, d_Y, BATCH, O, H);

    // 添加输出偏置 B2
    blocksPerGrid = (BATCH * O + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(add_bias_kernel, blocksPerGrid, threadsPerBlock, 0, 0,
                       d_Y, d_B2, BATCH, O);

    // 在最后一个 kernel 后记录 GPU 停止时间
    hipEventRecord(gpu_stop, 0);
    hipEventSynchronize(gpu_stop);

    // 计算 GPU 时间（毫秒）
    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, gpu_start, gpu_stop);

    // 销毁 GPU 事件
    hipEventDestroy(gpu_start);
    hipEventDestroy(gpu_stop);

    // 设备 -> 主机
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // 确保所有 GPU 操作完成，并记录 CPU 停止时间
    hipDeviceSynchronize();
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    // 打印部分输出
    std::cout << "MLP Forward Pass Results:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "Output[" << i << "]: ";
        for (int j = 0; j < O; ++j) {
            std::cout << h_Y[i * O + j] << " ";
        }
        std::cout << "\n";
    }

    // 打印计时
    std::cout << "GPU Computation Time: " << gpu_ms   << " ms\n";
    std::cout << "CPU Total Forward Time:  " << cpu_ms.count() << " ms\n";

    // 释放资源
    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    return 0;
}
