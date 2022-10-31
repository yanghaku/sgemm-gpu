#ifndef SGEMM_GPU_TIME_HELPER_H
#define SGEMM_GPU_TIME_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

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
    gettimeofday(&(t0), (void *)0);

#define GET_DURING(t1, t0) ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)
#define TO_SEED(t0) (t0.tv_sec * 1000 + t0.tv_usec)

#endif // _MSC_VER

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // SGEMM_GPU_TIME_HELPER_H
