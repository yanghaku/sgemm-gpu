#include "time_helper.h"

static inline void gemm_nn(unsigned int M, unsigned int N, unsigned int K, float ALPHA, const float *A,
                           unsigned int lda, const float *B, unsigned int ldb, float *C, unsigned int ldc) {
    unsigned int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static inline void gemm_nt(unsigned int M, unsigned int N, unsigned int K, float ALPHA, const float *A,
                           unsigned int lda, const float *B, unsigned int ldb, float *C, unsigned int ldc) {
    unsigned int i, j, k;
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

static inline void gemm_tn(unsigned int M, unsigned int N, unsigned int K, float ALPHA, const float *A,
                           unsigned int lda, const float *B, unsigned int ldb, float *C, unsigned int ldc) {
    unsigned int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

static inline void gemm_tt(unsigned int M, unsigned int N, unsigned int K, float ALPHA, const float *A,
                           unsigned int lda, const float *B, unsigned int ldb, float *C, unsigned int ldc) {
    unsigned int i, j, k;
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

double gemm_cpu(int TA, int TB, unsigned int M, unsigned int N, unsigned int K, float ALPHA, const float *A,
                unsigned int lda, const float *B, unsigned int ldb, float BETA, float *C, unsigned int ldc) {
    SET_TIME(t0)

    unsigned int i, j;
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
