#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward

#define BATCH 1024
#define I 10
#define H 20
#define O 5

// 矩阵乘法核函数：C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    return;
}

// 添加偏置核函数
__global__ void add_bias_kernel(double* mat, const double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        mat[idx] += bias[col];
    }
    return;
}

// ReLU激活函数核函数
__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
    return;
}

// 随机初始化函数
void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
    return;
}

int main() {
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H(BATCH * H), h_Y(BATCH * O);

    // 随机初始化所有数据
    srand(42); // 设置随机种子以便结果可重现
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // GPU内存分配
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    
    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H * sizeof(double));
    hipMalloc(&d_B1, H * sizeof(double));
    hipMalloc(&d_H, BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    // 将数据从主机复制到设备
    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize;

    // 隐藏层计算: H = X * W1
    // X: BATCH x I, W1: I x H, H: BATCH x H
    gridSize.x = (H + blockSize.x - 1) / blockSize.x;
    gridSize.y = (BATCH + blockSize.y - 1) / blockSize.y;
    hipLaunchKernelGGL(matmul_kernel, gridSize, blockSize, 0, 0, d_X, d_W1, d_H, BATCH, H, I);

    // 添加偏置 B1
    int threadsPerBlock = 256;
    int blocksPerGrid = (BATCH * H + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(add_bias_kernel, blocksPerGrid, threadsPerBlock, 0, 0, d_H, d_B1, BATCH, H);

    // 应用ReLU激活函数
    hipLaunchKernelGGL(relu_kernel, blocksPerGrid, threadsPerBlock, 0, 0, d_H, BATCH * H);

    // 输出层计算: Y = H * W2
    // H: BATCH x H, W2: H x O, Y: BATCH x O
    gridSize.x = (O + blockSize.x - 1) / blockSize.x;
    gridSize.y = (BATCH + blockSize.y - 1) / blockSize.y;
    hipLaunchKernelGGL(matmul_kernel, gridSize, blockSize, 0, 0, d_H, d_W2, d_Y, BATCH, O, H);

    // 添加输出层偏置 B2
    blocksPerGrid = (BATCH * O + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(add_bias_kernel, blocksPerGrid, threadsPerBlock, 0, 0, d_Y, d_B2, BATCH, O);

    // 将结果从设备复制回主机
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // 等待所有GPU计算完成
    hipDeviceSynchronize();

    // 打印一些输出值进行验证
    std::cout << "MLP Forward Pass Results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Output[" << i << "]: ";
        for (int j = 0; j < O; ++j)
            std::cout << h_Y[i * O + j] << " ";
        std::cout << std::endl;
    }

    // 释放GPU内存
    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    std::cout << "MLP forward pass completed successfully!" << std::endl;
    return 0;
}