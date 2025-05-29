#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <mpi.h>

constexpr int BLOCK_SIZE = 64;

template<typename T>
inline T minv(T a, T b) { return a < b ? a : b; }

// 初始化矩阵，随机填充
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
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

// Block tiling cache 优化
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         int N, int M, int P,
                         int block_size = BLOCK_SIZE) {
    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                int i_end = minv(ii + block_size, N);
                int j_end = minv(jj + block_size, P);
                int k_end = minv(kk + block_size, M);
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

// MPI + Cache Blocking
void matmul_mpi(const std::vector<double>& A,
                const std::vector<double>& B,
                std::vector<double>& C,
                int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_N = N / size;
    std::vector<double> local_A(local_N * M);
    std::vector<double> local_C(local_N * P, 0.0);

    MPI_Scatter(A.data(), local_N * M, MPI_DOUBLE,
                local_A.data(), local_N * M, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<double*>(B.data()),
              M * P, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

    // 本地计算使用 cache blocking
    for (int ii = 0; ii < local_N; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < M; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < P; jj += BLOCK_SIZE) {
                int i_max = minv(ii + BLOCK_SIZE, local_N);
                int k_max = minv(kk + BLOCK_SIZE, M);
                int j_max = minv(jj + BLOCK_SIZE, P);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double a = local_A[i * M + k];
                        for (int j = jj; j < j_max; ++j) {
                            local_C[i * P + j] += a * B[k * P + j];
                        }
                    }
                }
            }
        }
    }

    MPI_Gather(local_C.data(), local_N * P, MPI_DOUBLE,
               C.data(), local_N * P, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[MPI] Matrix multiplication completed." << std::endl;
    }
}

// Transpose + Cache Blocking
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C,
                  int N, int M, int P) {
    std::vector<double> Bt(P * M);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < P; ++j)
            Bt[j * M + i] = B[i * P + j];

    std::fill(C.begin(), C.end(), 0.0);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < P; jj += BLOCK_SIZE) {
            int i_max = minv(ii + BLOCK_SIZE, N);
            int j_max = minv(jj + BLOCK_SIZE, P);
            for (int i = ii; i < i_max; ++i) {
                for (int j = jj; j < j_max; ++j) {
                    double sum = 0.0;
                    int baseA = i * M;
                    int baseBt = j * M;
                    for (int k = 0; k < M; ++k) {
                        sum += A[baseA + k] * Bt[baseBt + k];
                    }
                    C[i * P + j] = sum;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    const int RUNS = 30;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        double total_ms = 0;
        for (int iter = 0; iter < RUNS; ++iter) {
            std::fill(C.begin(), C.end(), 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            matmul_mpi(A, B, C, N, M, P);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (rank == 0) total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        if (rank == 0) {
            std::cout << "[MPI] Valid: " << (validate(C, C_ref, N, P) ? "true" : "false")
                      << "  AvgTime: " << total_ms / RUNS << " ms\n";
        }
        MPI_Finalize();
    } else {
        double total_ms = 0;
        for (int iter = 0; iter < RUNS; ++iter) {
            std::fill(C.begin(), C.end(), 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            if (mode == "baseline") matmul_baseline(A, B, C, N, M, P);
            else if (mode == "openmp") matmul_openmp(A, B, C, N, M, P);
            else if (mode == "block") matmul_block_tiling(A, B, C, N, M, P, BLOCK_SIZE);
            else if (mode == "other") matmul_other(A, B, C, N, M, P);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::cout << "[" << mode << "] Valid: " << (validate(C, C_ref, N, P) ? "true" : "false")
                  << "  AvgTime: " << total_ms / RUNS << " ms\n";
    }

    return 0;
}
