#include "sgemm_gpu.h"
#include "utils.h"
#include <cublas_v2.h>

#ifdef RUN_CPU
extern double gemm_cpu(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, const float *A, size_t lda,
                       const float *B, size_t ldb, float BETA, float *C, size_t ldc);
#endif // RUN_CPU

double gemm_cublas(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, float *A_gpu, size_t lda, float *B_gpu,
                   size_t ldb, float BETA, float *C_gpu, size_t ldc) {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t0)
    CUBLAS_CALL(cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA,
                            B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc));
    CUDA_CALL(cudaDeviceSynchronize());
    SET_TIME(t1)

    CUBLAS_CALL(cublasDestroy(handle));
    return GET_DURING(t1, t0);
}

double call_gemm_gpu(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, float *A_gpu, size_t lda, float *B_gpu,
                     size_t ldb, float BETA, float *C_gpu, size_t ldc) {

    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t0)
    sgemm_gpu(TA, TB, (int)M, (int)N, (int)K, ALPHA, A_gpu, (int)lda, B_gpu, (int)ldb, BETA, C_gpu, (int)ldc);
    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t1)
    return GET_DURING(t1, t0);
}

constexpr auto cpu_check_eps = 1e-2;
constexpr auto gpu_check_eps = 1e-4;

void do_test(int TA, int TB, size_t m, size_t k, size_t n) {
    std::cerr << "Test FLOP = " << 2 * m * k * n << std::endl;
    std::cerr << "M = " << m << "; K = " << k << "; N = " << n << std::endl;
    SET_TIME(time_0)
    std::default_random_engine g(TO_SEED(time_0));
    std::uniform_real_distribution<float> d(-1, 1);

    auto count = m * n;
    auto bytes = count * sizeof(float);

    float alpha = d(g);
    float beta = d(g);
    auto lda = (!TA) ? k : m;
    auto ldb = (!TB) ? n : k;

    auto a = new float[m * k];
    auto b = new float[k * n];
    auto out_c = new float[m * n];
    randomize_matrix(a, m * k);
    randomize_matrix(b, k * n);
    auto gpu_a = create_cuda_matrix(a, m * k);
    auto gpu_b = create_cuda_matrix(b, k * n);
    auto gpu_c = create_cuda_matrix(nullptr, m * n);

    // compute with cublas
    auto cublas_out_c = new float[m * n];
    auto time_cublas = gemm_cublas(TA, TB, m, n, k, alpha, gpu_a, lda, gpu_b, ldb, beta, gpu_c, n);
    std::cerr << "Time cublas = \t" << time_cublas << " ms" << std::endl;
    CUDA_CALL(cudaMemcpy(cublas_out_c, gpu_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    // compute with cpu
#ifdef RUN_CPU
    memset(out_c, 0, bytes);
    auto time_cpu = gemm_cpu(TA, TB, m, n, k, alpha, a, lda, b, ldb, beta, out_c, n);
    std::cerr << "Time CPU = \t" << time_cpu << " ms" << std::endl;

    auto cpu_err_id = check_matrix(cublas_out_c, out_c, count, cpu_check_eps);
    if (cpu_err_id != count) {
        std::cerr << "Error: expect cpu_c[" << cpu_err_id << "] = " << std::fixed << cublas_out_c[cpu_err_id]
                  << " but get " << std::fixed << out_c[cpu_err_id] << std::endl;
    }

#endif // RUN_CPU

    // compute with kernel
    CUDA_CALL(cudaMemset(gpu_c, 0, bytes));
    CUDA_CALL(cudaDeviceSynchronize());
    auto time_my_kernel = call_gemm_gpu(TA, TB, m, n, k, alpha, gpu_a, lda, gpu_b, ldb, beta, gpu_c, n);
    std::cerr << "Time my = \t" << time_my_kernel << " ms" << std::endl;

    CUDA_CALL(cudaMemcpy(out_c, gpu_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    auto my_kernel_err_id = check_matrix(cublas_out_c, out_c, count, gpu_check_eps);
    for (auto i = 0; i < count; ++i) {
        if (fabs(cublas_out_c[i] - out_c[i]) > gpu_check_eps) {
            std::cerr << i << " " << i / n << "," << i % n << std::endl;
        }
    }
    if (my_kernel_err_id != count) {
        std::cerr << "Error: expect my_c[" << my_kernel_err_id << "] = " << std::fixed << cublas_out_c[my_kernel_err_id]
                  << " but get " << std::fixed << out_c[my_kernel_err_id] << std::endl;
        std::exit(-1);
    }

    // free all
    CUDA_CALL(cudaFree(gpu_c));
    CUDA_CALL(cudaFree(gpu_b));
    CUDA_CALL(cudaFree(gpu_a));
    delete[] cublas_out_c;
    delete[] out_c;
    delete[] b;
    delete[] a;
    std::cerr << "--------------------------------------" << std::endl << std::endl;
}

int main() {
    CUDA_CALL(cudaSetDevice(0));
    std::cout.precision(17);

    do_test(0, 0, 32, 1, 32);
    do_test(0, 0, 512, 5, 64);
    do_test(0, 0, 1024, 8, 32);
    do_test(0, 0, 64, 68, 512);
    do_test(0, 0, 1024, 1, 512);
    do_test(0, 0, 256, 5, 512);
    do_test(0, 0, 128, 8, 128);
    do_test(0, 0, 256, 12, 512);
    do_test(0, 0, 256, 1028, 128);
    do_test(0, 0, 128, 128, 256 + 64 + 32);
    do_test(0, 0, 1024 + 64, 1024, 1024);
    do_test(0, 0, 2048, 2048, 2048);
    do_test(0, 0, 4096, 4096, 4096);
    //    do_test(0, 0, 64, 2916, 363);
    //    do_test(0, 0, 192, 729, 1600);
    //    do_test(0, 0, 384, 196, 1728);
    //    do_test(0, 0, 256, 196, 3456);
    //    do_test(0, 0, 256, 196, 2304);
    //    do_test(0, 0, 128, 4096, 12544);
    //    do_test(0, 0, 128, 4096, 4096);
    //    do_test(0, 0, 64, 75, 12544);
    //    do_test(0, 0, 64, 75, 12544);
    //    do_test(0, 0, 64, 75, 12544);
    //    do_test(0, 0, 64, 576, 12544);
    //    do_test(0, 0, 256, 2304, 784);
    //    do_test(1, 1, 2304, 256, 784);
    //    do_test(0, 0, 512, 4608, 196);
    //    do_test(1, 1, 4608, 512, 196);

    CUDA_CALL(cudaDeviceReset());
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

/// these codes are copy from darknet

#ifdef RUN_CPU
static inline void gemm_nn(size_t M, size_t N, size_t K, float ALPHA, const float *A, size_t lda, const float *B,
                           size_t ldb, float *C, size_t ldc) {
    size_t i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static inline void gemm_nt(size_t M, size_t N, size_t K, float ALPHA, const float *A, size_t lda, const float *B,
                           size_t ldb, float *C, size_t ldc) {
    size_t i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static inline void gemm_tn(size_t M, size_t N, size_t K, float ALPHA, const float *A, size_t lda, const float *B,
                           size_t ldb, float *C, size_t ldc) {
    size_t i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static inline void gemm_tt(size_t M, size_t N, size_t K, float ALPHA, const float *A, size_t lda, const float *B,
                           size_t ldb, float *C, size_t ldc) {
    size_t i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

double gemm_cpu(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, const float *A, size_t lda, const float *B,
                size_t ldb, float BETA, float *C, size_t ldc) {
    SET_TIME(t0)

    size_t i, j;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);

    SET_TIME(t1)
    return GET_DURING(t1, t0);
}
#endif // RUN_CPU
