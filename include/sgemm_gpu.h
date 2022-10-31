#ifndef SGEMM_GPU_SGEMM_GPU_H
#define SGEMM_GPU_SGEMM_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

void sgemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb,
               float BETA, float *C_gpu, int ldc);

#ifdef __cplusplus
}
#endif

#endif // SGEMM_GPU_SGEMM_GPU_H
