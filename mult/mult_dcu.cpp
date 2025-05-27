#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>      // 新增
#include <omp.h>
// #include <mpi.h>
#include <hip/hip_runtime.h>
// 在文件开头定义块大小
constexpr int TILE_SIZE = 16;
// 简单的错误检查
inline void hipCheck(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << msg
                  << " (" << hipGetErrorString(err) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}



// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp

// 运行 baseline
// ./outputfile baseline

// 运行 OpenMP
// ./outputfile openmp

// 运行 子块并行优化
// ./outputfile block

// 运行 MPI（假设 4 个进程）
// mpirun -np 4 ./outputfile mpi

// 运行 MPI（假设 4 个进程）
// ./outputfile other
// 初始化矩阵（以一维数组形式表示），随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证优化后结果与baseline是否一致
bool validate(const std::vector<double>& A,
              const std::vector<double>& B,
              int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础矩阵乘法baseline实现
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C,
                     int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    }
}

// 方式1: OpenMP多线程并行
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int N, int M, int P) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    }
}

// 方式2: 子块（block）并行优化
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         int N, int M, int P,
                         int block_size = 64) {
    // 初始化输出
    std::fill(C.begin(), C.end(), 0.0);

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                int i_max = std::min(ii + block_size, N);
                int k_max = std::min(kk + block_size, M);
                int j_max = std::min(jj + block_size, P);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double a_val = A[i * M + k];
                        for (int j = jj; j < j_max; ++j) {
                            C[i * P + j] += a_val * B[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
// void matmul_mpi(int N, int M, int P) 
// {
//     std::cout << "matmul_mpi methods..." << std::endl;
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int local_N = N / size;
//     std::vector<double> local_A(local_N * M);
//     std::vector<double> B(M * P);
//     std::vector<double> local_C(local_N * P, 0);
//     std::vector<double> C;

//     if (rank == 0) 
//     {
//         std::vector<double> A(N * M);
//         init_matrix(A, N, M);
//         init_matrix(B, M, P);
//         C.resize(N * P);
//         MPI_Scatter(A.data(), local_N * M, MPI_DOUBLE, local_A.data(), local_N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     } 
//     else 
//     {
//         MPI_Scatter(nullptr, local_N * M, MPI_DOUBLE, local_A.data(), local_N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     }

//     MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     for (int i = 0; i < local_N; ++i)
//     {
//         for (int j = 0; j < P; ++j) 
//         {
//             double sum = 0;
//             for (int k = 0; k < M; ++k)
//             {
//                 sum += local_A[i * M + k] * B[k * P + j];
//             }
//             local_C[i * P + j] = sum;
//         }
//     }

//     MPI_Gather(local_C.data(), local_N * P, MPI_DOUBLE,rank == 0 ? C.data() : nullptr, local_N * P, MPI_DOUBLE,0, MPI_COMM_WORLD);

//     if (rank == 0)
//     {
//         std::cout << "[MPI] Matrix multiplication completed." << std::endl;
//     }
// }


// 方式4: 矩阵转置优化（Other）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int N, int M, int P) {
    // 转置B -> Bt (P x M)
    std::vector<double> Bt(P * M);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            Bt[j * M + i] = B[i * P + j];

    // 并行乘法
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * Bt[j * M + k];
            C[i * P + j] = sum;
        }
    }
}

// DCU核函数：矩阵乘法（分块优化版本）
__global__ void matmul_tiled_kernel(const double* A, const double* B, double* C,
                                    int N, int M, int P) {
    // 共享内存声明
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

    for (int k = 0; k < M; k += TILE_SIZE) {
        // 从全局内存加载数据到共享内存
        if (row < N && (k + tx) < M) {
            As[ty][tx] = A[row * M + k + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if ((k + ty) < M && col < P) {
            Bs[ty][tx] = B[(k + ty) * P + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        // 计算部分和
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    // 写入结果
    if (row < N && col < P) {
        C[row * P + col] = sum;
    }
}

// Host 端封装：分配、拷贝、执行、回拷、释放
void matmul_dcu(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C,
                int N, int M, int P) {
    double *d_A, *d_B, *d_C;
    size_t size_A = N * M * sizeof(double);
    size_t size_B = M * P * sizeof(double);
    size_t size_C = N * P * sizeof(double);

    // 1. 分配设备内存
    hipCheck(hipMalloc(&d_A, size_A), "hipMalloc A");
    hipCheck(hipMalloc(&d_B, size_B), "hipMalloc B");
    hipCheck(hipMalloc(&d_C, size_C), "hipMalloc C");

    // 2. 拷贝数据到设备
    hipCheck(hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice), "hipMemcpy A");
    hipCheck(hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice), "hipMemcpy B");

    // 3. 配置核函数参数
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // 4. 启动核函数
    hipLaunchKernelGGL(matmul_tiled_kernel, 
                       gridDim, blockDim, 0, 0, 
                       d_A, d_B, d_C, N, M, P);
    hipCheck(hipGetLastError(), "kernel launch");

    // 5. 同步设备
    hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");

    // 6. 拷贝结果回主机
    hipCheck(hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost), "hipMemcpy C");

    // 7. 释放设备内存
    hipCheck(hipFree(d_A), "hipFree A");
    hipCheck(hipFree(d_B), "hipFree B");
    hipCheck(hipFree(d_C), "hipFree C");
}





int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    // if (mode == "mpi") {
    //     MPI_Init(&argc, &argv);

    //     int rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //     // 计时开始
    //     auto t0 = std::chrono::high_resolution_clock::now();

    //     matmul_mpi(A, B, C, N, M, P);

    //     // 计时结束
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    //     if (rank == 0) {
    //         bool ok = validate(C, C_ref, N, P);
    //         std::cout << "[MPI] Valid: " << (ok ? "true" : "false")
    //                   << "  Time: " << ms << " ms" << std::endl;
    //     }

    //     MPI_Finalize();
    // }
        if (mode == "baseline") {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "[Baseline] Done.";
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  Time: " << ms << " ms" << std::endl;
    }
    else if (mode == "openmp") {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "[OpenMP] Valid: "
                  << (validate(C, C_ref, N, P) ? "true" : "false")
                  << "  Time: " << ms << " ms" << std::endl;
    }
    else if (mode == "block") {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_block_tiling(A, B, C, N, M, P);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "[Block] Valid: "
                  << (validate(C, C_ref, N, P) ? "true" : "false")
                  << "  Time: " << ms << " ms" << std::endl;
    }
    else if (mode == "other") {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "[Other] Valid: "
                  << (validate(C, C_ref, N, P) ? "true" : "false")
                  << "  Time: " << ms << " ms" << std::endl;
    }
    else if (mode == "dcu") {
        // 计时开始
        auto t0 = std::chrono::high_resolution_clock::now();

        // 调用 DCU 版本
        matmul_dcu(A, B, C, N, M, P);

        // 计时结束
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 验证结果
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[DCU] Valid: " << (ok ? "true" : "false")
                  << "  Time: " << ms << " ms" << std::endl;
    }
   else {
        std::cerr << "Usage: ./outputfile [baseline|openmp|block|other|dcu]" 
                  << std::endl;
    }
    return 0;
}