#ifndef CUBLAS_GEMM_UTILS_H
#define CUBLAS_GEMM_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <random>

#ifdef _MSC_VER

#include <Windows.h>

#define SET_TIME(t0)                                                                                                   \
    long long(t0);                                                                                                     \
    GetSystemTimePreciseAsFileTime((LPFILETIME)(&(t0)));

#define GET_DURING(t1, t0) (((double)((t1) - (t0))) / 10000.0)
#define TO_SEED(t0) t0

#else // _MSC_VER

#include <sys/time.h>

#define SET_TIME(t0)                                                                                                   \
    struct timeval(t0);                                                                                                \
    gettimeofday(&(t0), nullptr);

#define GET_DURING(t1, t0) ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)
#define TO_SEED(t0) (t0.tv_sec * 1000 + t0.tv_usec)

#endif // _MSC_VER

#define CHECK_OR_REPORT_CUDA_ERROR(err, call_stat_str)                                                                 \
    do {                                                                                                               \
        auto err_code = (err);                                                                                         \
        if (err_code != cudaSuccess) {                                                                                 \
            std::cerr << "CUDA Error At " __FILE__ ":" << __LINE__ << "; " call_stat_str << std::endl;                 \
            std::cerr << "Error Name=" << cudaGetErrorName(err_code)                                                   \
                      << ", Error Message=" << cudaGetErrorString(err_code) << std::endl;                              \
            std::exit(err_code);                                                                                       \
        }                                                                                                              \
    } while (0)

#define CUDA_CALL(...)                                                                                                 \
    do {                                                                                                               \
        CHECK_OR_REPORT_CUDA_ERROR(cudaPeekAtLastError(), "before " #__VA_ARGS__);                                     \
        __VA_ARGS__;                                                                                                   \
        CHECK_OR_REPORT_CUDA_ERROR(cudaGetLastError(), "after " #__VA_ARGS__);                                         \
    } while (0)

#define CUBLAS_CALL(...)                                                                                               \
    do {                                                                                                               \
        auto res = (__VA_ARGS__);                                                                                      \
        if (res != CUBLAS_STATUS_SUCCESS) {                                                                            \
            std::cerr << "CUBLAS Error At " __FILE__ ":" << __LINE__ << #__VA_ARGS__ << std::endl;                     \
            std::cerr << "Error Code = " << res << std::endl;                                                          \
            std::exit(res);                                                                                            \
        }                                                                                                              \
    } while (0)

inline void randomize_matrix(float *m, size_t n) {
    SET_TIME(t)
    std::default_random_engine generator(TO_SEED(t));
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (auto i = 0; i < n; ++i) {
        m[i] = distribution(generator);
    }
}

inline size_t check_matrix(float *m1, float *m2, size_t n, double eps) {
    for (auto i = 0; i < n; ++i) {
        if (fabs(double(m1[i]) - double(m2[i])) > eps) {
            return i;
        }
    }
    return n;
}

inline float *create_cuda_matrix(float *m, size_t num) {
    float *out = nullptr;
    size_t bytes = sizeof(float) * num;
    CUDA_CALL(cudaMalloc(&out, bytes));
    if (m == nullptr) {
        CUDA_CALL(cudaMemset(out, 0, bytes));
    } else {
        CUDA_CALL(cudaMemcpy(out, m, bytes, cudaMemcpyHostToDevice));
    }
    return out;
}

#endif // CUBLAS_GEMM_UTILS_H
