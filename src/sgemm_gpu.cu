#include "sgemm_kernels/sgemm_kernels.cuh"
#include <cstdio>
#include <cstdlib>

#define NOT_IMPLEMENTED(...)                                                                                           \
    do {                                                                                                               \
        fprintf(stderr, "The condition '" #__VA_ARGS__ "' has not been implemented in function '%s' line %d\n",        \
                __func__, __LINE__);                                                                                   \
        std::exit(-1);                                                                                                 \
    } while (0)

static __host__ always_inline void sgemm_nn_host(unsigned int M, unsigned int N, unsigned int K, float alpha,
                                                 const float *A_gpu, unsigned int lda, const float *B_gpu,
                                                 unsigned int ldb, float beta, float *C_gpu, unsigned int ldc) {
    /*
     * ---------------------------------------------
     * |                         |             |   |
     * |                         |             |   |
     * |     A: 128x128          |  B: 32x32   |   |
     * |                         |             |   |
     * |                         |             | D |
     * |                         |             |   |
     * |-------------------------|             |   |
     * |                         |             |   |
     * |     C: 32x32            |             |   |
     * |                         |             |   |
     * |-------------------------|-------------|   |
     * |      E:  < 32                         |   |
     * |--------------------------------------------
     */
    if ((MOD4(ldb) == 0) && (MOD4(ldc) == 0) && ((MOD4(lda) == 0) || K < 8)) {
        dim3 grid_dim(1), block_dim(1);

        auto M_div128 = M >> 7;
        auto N_div128 = N >> 7;
        if (M_div128 > 0 && N_div128 > 0) {
            // A: 128x128
            grid_dim.x = M_div128;
            grid_dim.y = N_div128;
            block_dim.x = 256;
            sgemm_nn_128x128_batch_float4<<<grid_dim, block_dim>>>(K, alpha, A_gpu, lda, B_gpu, ldb, beta, C_gpu, ldc);
        }

        auto N_mod128 = MOD128(N);
        if (N_mod128) {
            auto M_div32 = M >> 5;
            auto N_mod128_div32 = N_mod128 >> 5;
            if (M_div32 > 0 && N_mod128_div32 > 0) {
                // B: 32x32
                grid_dim.x = M_div32;
                grid_dim.y = N_mod128_div32;
                block_dim.x = 64;
                auto col_offset = N_div128 << 7;
                sgemm_nn_32x32_batch_float4<<<grid_dim, block_dim>>>(K, alpha, A_gpu, lda, B_gpu + col_offset, ldb,
                                                                     beta, C_gpu + col_offset, ldc);
            }

            auto N_mod128_mod32 = MOD32(N_mod128);
            if (N_mod128_mod32) {
                // D: row=M, col<32
                NOT_IMPLEMENTED(0);
            }
        }

        auto M_mod128 = MOD128(M);
        if (M_mod128) {
            if (N_div128 > 0) {
                auto M_mod128_div32 = M_mod128 >> 5;
                if (M_mod128_div32 > 0) {
                    // C: 32x32
                    grid_dim.x = M_mod128_div32;
                    grid_dim.y = N_div128 << 2; // N_div128 * 128 / 32
                    block_dim.x = 64;
                    auto row_offset = M_div128 << 7;
                    sgemm_nn_32x32_batch_float4<<<grid_dim, block_dim>>>(K, alpha, A_gpu + row_offset * lda, lda, B_gpu,
                                                                         ldb, beta, C_gpu + row_offset * ldc, ldc);
                }
            }

            auto M_mod128_mod32 = MOD32(M_mod128);
            if (M_mod128_mod32) {
                // E: row<32, col=N-N%32
                NOT_IMPLEMENTED(0);
            }
        }
    } else {
        NOT_IMPLEMENTED("align is not 4");
    }
}

static __host__ always_inline void sgemm_tn_host(unsigned int M, unsigned int N, unsigned int K, float alpha,
                                                 const float *A_gpu, unsigned int lda, const float *B_gpu,
                                                 unsigned int ldb, float beta, float *C_gpu, unsigned int ldc) {
}

static __host__ always_inline void sgemm_nt_host(unsigned int M, unsigned int N, unsigned int K, float alpha,
                                                 const float *A_gpu, unsigned int lda, const float *B_gpu,
                                                 unsigned int ldb, float beta, float *C_gpu, unsigned int ldc) {
}

extern "C" void sgemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, const float *A_gpu, int lda,
                          const float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc) {
    fprintf(stderr,
            "run %s(TA=%d,TB=%d,M=%d,N=%d,K=%d,ALPHA=%f,A_GPU=%p,lda=%d,B_GPU=%p,ldb=%d,BETA=%f,C_gpu=%p,ldc=%d)\n",
            __func__, TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);
    return;
    if (!(M > 0 && N > 0 && K > 0 && lda > 0 && ldb > 0 && ldc > 0)) {
        return;
    }

    if (!TA && !TB) {
        sgemm_nn_host(M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);
    } else if (TA && !TB) {
        sgemm_tn_host(M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);
    } else if (!TA) {
        sgemm_nt_host(M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);
    } else {
        NOT_IMPLEMENTED(TA == 1 && TB == 1);
    }
}
