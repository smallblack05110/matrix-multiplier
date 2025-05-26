#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

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

void matmul_mpi(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C,
                int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. 划分行数
    int base = N / size;
    int rem  = N % size;
    std::vector<int> rows(size);
    for (int i = 0; i < size; ++i)
        rows[i] = base + (i < rem ? 1 : 0);

    // 2. 计算 Scatterv 参数
    std::vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = rows[i] * M;
        displs[i]     = offset;
        offset       += sendcounts[i];
    }

    // 3. Scatter A 的本地块
    int local_rows = rows[rank];
    std::vector<double> A_local(local_rows * M);
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), sendcounts[rank],      MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 4. Broadcast B
    std::vector<double> B_full;
    if (rank == 0)       B_full = B;
    else                 B_full.resize(M * P);
    MPI_Bcast(B_full.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 5. 本地 compute
    std::vector<double> C_local(local_rows * P, 0.0);
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A_local[i * M + k] * B_full[k * P + j];
            C_local[i * P + j] = sum;
        }
    }

    // 6. Gather 到 rank 0
    std::vector<int> recvcounts(size), recvdispls(size);
    offset = 0;
    for (int i = 0; i < size; ++i) {
        recvcounts[i]  = rows[i] * P;
        recvdispls[i]  = offset;
        offset        += recvcounts[i];
    }

    if (rank == 0)
        C.assign(N * P, 0.0);
    MPI_Gatherv(C_local.data(), local_rows * P, MPI_DOUBLE,
                C.data(),        recvcounts.data(), recvdispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 7. 新增：把完整 C 广播给所有进程
    if (rank != 0)
        C.assign(N * P, 0.0); 
    MPI_Bcast(C.data(), N * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


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

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        matmul_mpi(A, B, C, N, M, P);
        MPI_Finalize();
        std::cout << "[MPI] Valid: " << (validate(C, C_ref, N, P) ? "true" : "false") << std::endl;
    }
    else if (mode == "baseline") {
        std::cout << "[Baseline] Done." << std::endl;
    }
    else if (mode == "openmp") {
        matmul_openmp(A, B, C, N, M, P);
        std::cout << "[OpenMP] Valid: " << (validate(C, C_ref, N, P) ? "true" : "false") << std::endl;
    }
    else if (mode == "block") {
        matmul_block_tiling(A, B, C, N, M, P);
        std::cout << "[Block] Valid: " << (validate(C, C_ref, N, P) ? "true" : "false") << std::endl;
    }
    else if (mode == "other") {
        matmul_other(A, B, C, N, M, P);
        std::cout << "[Other] Valid: " << (validate(C, C_ref, N, P) ? "true" : "false") << std::endl;
    }
    else {
        std::cerr << "Usage: ./outputfile [baseline|openmp|block|mpi|other]" << std::endl;
    }
    return 0;
}
