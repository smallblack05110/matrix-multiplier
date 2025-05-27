#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <hip/hip_runtime.h>

// 在文件开头定义块大小
enum { TILE_SIZE = 16 };

// 简单的错误检查
inline void hipCheck(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << msg
                  << " (" << hipGetErrorString(err) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

// 初始化矩阵并随机填充
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i) mat[i] = dist(gen);
}

// 验证结果
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
                         int block_size /*=64*/) 
{
    // 外层两层循环并行，每个线程处理一个小块
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            // 对当前 (ii, jj) 块，遍历所有 kk 块
            for (int kk = 0; kk < M; kk += block_size) {
                int i_end = std::min(ii + block_size, N);
                int j_end = std::min(jj + block_size, P);
                int k_end = std::min(kk + block_size, M);

                // 块内部标准三层循环
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        double a_val = A[i * M + k];
                        for (int j = jj; j < j_end; ++j) {
                            C[i * P + j] += a_val * B[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

// // 方式3: MPI 多进程并行
// void matmul_mpi(const std::vector<double>& A,
//                 const std::vector<double>& B,
//                 std::vector<double>& C,
//                 int N, int M, int P)
// {
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int local_N = N / size;
//     std::vector<double> local_A(local_N * M);
//     std::vector<double> local_C(local_N * P, 0.0);

//     MPI_Scatter(A.data(), local_N * M, MPI_DOUBLE,
//                 local_A.data(), local_N * M, MPI_DOUBLE,
//                 0, MPI_COMM_WORLD);

//     MPI_Bcast(const_cast<double*>(B.data()),
//               M * P, MPI_DOUBLE,
//               0, MPI_COMM_WORLD);

//     for (int i = 0; i < local_N; ++i) {
//         for (int j = 0; j < P; ++j) {
//             double sum = 0.0;
//             for (int k = 0; k < M; ++k) {
//                 sum += local_A[i * M + k] * B[k * P + j];
//             }
//             local_C[i * P + j] = sum;
//         }
//     }

//     MPI_Gather(local_C.data(), local_N * P, MPI_DOUBLE,
//                C.data(), local_N * P, MPI_DOUBLE,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::cout << "[MPI] Matrix multiplication completed." << std::endl;
//     }
// }

// 方式4: 矩阵转置优化（Other）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int N, int M, int P) {
    std::vector<double> Bt(P * M);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            Bt[j * M + i] = B[i * P + j];

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

// DCU 核函数
__global__ void matmul_tiled_kernel(const double* A, const double* B, double* C,
                                    int N, int M, int P) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    double sum = 0.0;
    for (int k = 0; k < M; k += TILE_SIZE) {
        As[ty][tx] = (row < N && k+tx < M) ? A[row*M + k + tx] : 0.0;
        Bs[ty][tx] = (k+ty < M && col < P) ? B[(k+ty)*P + col] : 0.0;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < N && col < P) C[row*P + col] = sum;
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";
    std::vector<double> A(N*M), B(M*P), C(N*P), C_ref(N*P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    double *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    size_t size_A=N*M*sizeof(double), size_B=M*P*sizeof(double), size_C=N*P*sizeof(double);

    // —— 包含分配在内的预处理计时 ——
    auto alloc_t0 = std::chrono::high_resolution_clock::now();
    hipCheck(hipMalloc(&d_A, size_A), "hipMalloc A");
    hipCheck(hipMalloc(&d_B, size_B), "hipMalloc B");
    hipCheck(hipMalloc(&d_C, size_C), "hipMalloc C");
    hipCheck(hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice), "hipMemcpy A");
    hipCheck(hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice), "hipMemcpy B");
    auto alloc_t1 = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();
    std::cout << "[DCU alloc+H2D] Time: " << alloc_ms << " ms" << std::endl;

    if (mode == "dcu") {
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        hipEventRecord(start, nullptr);

        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((P+TILE_SIZE-1)/TILE_SIZE,(N+TILE_SIZE-1)/TILE_SIZE);
        hipLaunchKernelGGL(matmul_tiled_kernel,
                           gridDim, blockDim, 0, 0,
                           d_A, d_B, d_C, N, M, P);

        hipEventRecord(stop, nullptr);
        hipEventSynchronize(stop);
        float kernel_ms = 0;
        hipEventElapsedTime(&kernel_ms, start, stop);
        std::cout << "[DCU-kernel] Time: " << kernel_ms << " ms" << std::endl;

        hipCheck(hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost), "hipMemcpy C");
        bool ok = validate(C, C_ref, N, P);
        std::cout << "[DCU] Valid: " << (ok?"true":"false") << std::endl;

        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    return 0;
}
