#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <mpi.h>

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
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                int i_end = std::min(ii + block_size, N);
                int j_end = std::min(jj + block_size, P);
                int k_end = std::min(kk + block_size, M);

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

// 方式3: MPI 多进程并行
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

    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += local_A[i * M + k] * B[k * P + j];
            }
            local_C[i * P + j] = sum;
        }
    }

    MPI_Gather(local_C.data(), local_N * P, MPI_DOUBLE,
               C.data(), local_N * P, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) std::cout << "[MPI] Matrix multiplication completed." << std::endl;
}

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

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";
    const int repeats = 50;

    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    auto run_and_time = [&](auto&& func, const std::string& tag) {
        double total_ms = 0.0;
        for (int r = 0; r < repeats; ++r) {
            std::fill(C.begin(), C.end(), 0.0);
            auto t0 = std::chrono::high_resolution_clock::now();
            func(A, B, C, N, M, P);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double avg_ms = total_ms / repeats;
        bool ok = validate(C, C_ref, N, P);
        int rank = 0;
        if (mode == "mpi") MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "[" << tag << "] Valid: " << (ok?"true":"false")
                      << "  Avg Time: " << avg_ms << " ms over " << repeats << " runs\n";
        }
    };

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        run_and_time(matmul_mpi, "MPI");
        MPI_Finalize();
    } else if (mode == "baseline") {
        run_and_time(matmul_baseline, "Baseline");
    } else if (mode == "openmp") {
        run_and_time(matmul_openmp, "OpenMP");
    } else if (mode == "block") {
        run_and_time(
            [&](auto&& A, auto&& B, auto&& C, int n, int m, int p) { matmul_block_tiling(A, B, C, n, m, p, 64); },
            "Block");
    } else if (mode == "other") {
        run_and_time(matmul_other, "Other");
    } else {
        std::cerr << "Usage: ./outputfile [baseline|openmp|block|mpi|other]\n";
    }
    return 0;
}
