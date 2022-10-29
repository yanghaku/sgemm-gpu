#ifndef CUBLAS_GEMM_KERNEL11
#define CUBLAS_GEMM_KERNEL11

#include <cstddef>

double gemm_gpu(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, float *A_gpu, size_t lda, float *B_gpu,
                size_t ldb, float BETA, float *C_gpu, size_t ldc);

#endif // CUBLAS_GEMM_KERNEL11
