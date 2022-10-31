#ifndef SGEMM_KERNELS_CUH
#define SGEMM_KERNELS_CUH

#ifdef _MSC_VER
#define always_inline __forceinline
#else
#define always_inline __always_inline
#endif // _MSC_VER

#define MOD2(val) ((val)&1)
#define MOD4(val) ((val)&3)
#define MOD8(val) ((val)&7)
#define MOD32(val) ((val)&31)
#define MOD128(val) ((val)&127)

#define IDX2F(row, col, ld) ((row) * (ld) + (col))
#define Matrix_addr(P, row, col, ld) (P + IDX2F(row, col, ld))

#define float4_ptr(ptr) ((float4 *)(ptr))
#define float4_val_ref(ptr) (*float4_ptr(ptr))
#define float4_load(v, ptr)                                                                                            \
    do {                                                                                                               \
        v = float4_val_ref(ptr);                                                                                       \
    } while (0)
#define float4_store(v, ptr)                                                                                           \
    do {                                                                                                               \
        float4_val_ref(ptr) = v;                                                                                       \
    } while (0)

#define float4_add_mul(f_v, f_v1, add)                                                                                 \
    do {                                                                                                               \
        f_v.x += f_v1.x * (add);                                                                                       \
        f_v.y += f_v1.y * (add);                                                                                       \
        f_v.z += f_v1.z * (add);                                                                                       \
        f_v.w += f_v1.w * (add);                                                                                       \
    } while (0)

#define float4_add_mul_add_mull(f_v, alpha, f_v1, beta, f_v2)                                                          \
    do {                                                                                                               \
        f_v.x = alpha * f_v1.x + beta * f_v2.x;                                                                        \
        f_v.y = alpha * f_v1.y + beta * f_v2.y;                                                                        \
        f_v.z = alpha * f_v1.z + beta * f_v2.z;                                                                        \
        f_v.w = alpha * f_v1.w + beta * f_v2.w;                                                                        \
    } while (0)

// kernels
__global__ void sgemm_nn_128x128_batch_float4(unsigned int K, float alpha, const float *A_gpu, unsigned int lda,
                                              const float *B_gpu, unsigned int ldb, float beta, float *C_gpu,
                                              unsigned int ldc);
__global__ void sgemm_nn_32x32_batch_float4(unsigned int K, float alpha, const float *A_gpu, unsigned int lda,
                                            const float *B_gpu, unsigned int ldb, float beta, float *C_gpu,
                                            unsigned int ldc);

#endif // SGEMM_KERNELS_CUH
