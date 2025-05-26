#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>      // 新增
#include <omp.h>
#include <mpi.h>

// … 省略 init_matrix, validate, matmul_baseline, matmul_openmp, matmul_block_tiling, matmul_mpi, matmul_other 的实现 …

int main(int argc, char** argv) {
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A, N, M);
    init_matrix(B, M, P);
    matmul_baseline(A, B, C_ref, N, M, P);

    if (mode == "mpi") {
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // 计时开始
        auto t0 = std::chrono::high_resolution_clock::now();

        matmul_mpi(A, B, C, N, M, P);

        // 计时结束
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0) {
            bool ok = validate(C, C_ref, N, P);
            std::cout << "[MPI] Valid: " << (ok ? "true" : "false")
                      << "  Time: " << ms << " ms" << std::endl;
        }

        MPI_Finalize();
    }
    else if (mode == "baseline") {
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
    else {
        std::cerr << "Usage: ./outputfile [baseline|openmp|block|mpi|other]" 
                  << std::endl;
    }
    return 0;
}
