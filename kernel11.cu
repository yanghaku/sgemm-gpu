#include "kernel11.h"
#include "utils.h"

#define block_dim() (dim3(256))
#define CEIL_DIV(size) ((((unsigned int)(size)) + 127) >> 7) // (size+128-1)/128)
#define grid_dim(M, N) (dim3(CEIL_DIV(M), CEIL_DIV(N)))
#undef CEIL_DIV

#define IDX2F(i, j, ld) ((i) + (j) * (ld))
#define Matrix_addr(P, i, j, ld) (P + IDX2F(i, j, ld))
#define Shared_addr(P, i, j) (P + (i + ((j) << 7)))

#define float4_val_ref(ptr) (*((float4 *)(ptr)))
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

__global__ __launch_bounds__(256) void sgemm_nn(unsigned int K, float alpha, const float *A_gpu, unsigned int lda,
                                                const float *B_gpu, unsigned int ldb, float beta, float *C_gpu,
                                                unsigned int ldc) {
    __shared__ float shared_a[256 * 4 * 2];
    __shared__ float shared_b[256 * 4 * 2];
    auto ptr_shared_a = shared_a;
    auto ptr_shared_b = shared_b;
    float4 A_v1[2], A_v2[2], B_v1[2], B_v2[2], C_v[16], C_res[16];
    memset(C_res, 0, sizeof(C_res));

    auto _tmp_bx_mul_128 = blockIdx.x << 7;
    auto _tmp_by_mul_128 = blockIdx.y << 7;
    // change point to sub matrix
    A_gpu += _tmp_by_mul_128 * lda;
    B_gpu += _tmp_bx_mul_128;
    C_gpu += _tmp_bx_mul_128 + _tmp_by_mul_128 * ldc;

    auto tx = threadIdx.x;
    // calculate the row and col
    auto _tmp_warp_id = tx >> 5;                           // tx / 32
    auto _tmp_lane_id = tx & 0x1f;                         // tx % 32
    auto _tmp_warp_row = _tmp_warp_id & 0x3;               // _tmp_warp_id % 4
    auto _tmp_warp_col = _tmp_warp_id >> 2;                // _tmp_warp_id / 4
    auto _tmp_row_w = _tmp_lane_id & 0x3;                  // _tmp_lane_id % 4
    auto _tmp_col_w = _tmp_lane_id >> 2;                   // _tmp_lane_id / 4
    auto row_c = (_tmp_warp_row << 5) + (_tmp_row_w << 3); // _tmp_warp_row * 32 + _tmp_row_w * 8;
    auto col_c = (_tmp_warp_col << 6) + (_tmp_col_w << 3); // _tmp_warp_col * 64 + _tmp_col_w * 8;

    // row and col for A, B to load from global matrix (load to shared memory)
    auto row_a = tx >> 1;       // tx / 2
    auto col_a = (tx & 1) << 2; // (tx % 2) * 4
    auto row_b = tx & 0x7;      // (tx % 8)
    auto col_b = tx >> 1;       // tx / 8 * 4

    // every thread load the corresponding value from global matrix to shared memory and sync
    ((float4 *)ptr_shared_b)[tx] = float4_val_ref(Matrix_addr(B_gpu, row_b, col_b, ldb));
    auto _tmp_global_a = float4_val_ref(Matrix_addr(A_gpu, row_a, col_a, lda));
    auto _tmp_shared_a_index = row_a + ((tx >> 3) << 5); // tx / 8 * 32 + tx % 8;
    ptr_shared_a[_tmp_shared_a_index] = _tmp_global_a.x;
    ptr_shared_a[_tmp_shared_a_index + 8] = _tmp_global_a.y;
    ptr_shared_a[_tmp_shared_a_index + 16] = _tmp_global_a.z;
    ptr_shared_a[_tmp_shared_a_index + 24] = _tmp_global_a.w;
    __syncthreads();

    auto lda8 = lda << 3;
    K >>= 3; // K = K / 8
    for (auto k_count = 1; k_count <= K; ++k_count) {
        // load from shared memory
        float4_load(A_v1[0], (ptr_shared_a + (row_c << 8)));              // ptr_shared_a[row_c*2*4]
        float4_load(A_v2[0], (ptr_shared_a + (((row_c << 1) | 1) << 2))); // ptr_shared_a[(row*2+1)*4]
        float4_load(B_v1[0], (ptr_shared_b + (row_c << 8)));              // ptr_shared_b[row_c*2*4]
        float4_load(B_v2[0], (ptr_shared_b + (((row_c << 1) | 1) << 2))); // ptr_shared_b[(row*2+1)*4]

        C_res[0].x += A_v1[0].x * A_v1[0].y;
#pragma unroll
        for (auto inner_k_count = 0; inner_k_count < 8; ++inner_k_count) {
            auto next_inner_k_count = (inner_k_count + 1) & 7;
            float4_load(A_v1[0], (ptr_shared_a + (row_c << 8)));              // ptr_shared_a[row_c*2*4]
            float4_load(A_v2[0], (ptr_shared_a + (((row_c << 1) | 1) << 2))); // ptr_shared_a[(row*2+1)*4]
            float4_load(B_v1[0], (ptr_shared_b + (row_c << 8)));              // ptr_shared_b[row_c*2*4]
            float4_load(B_v2[0], (ptr_shared_b + (((row_c << 1) | 1) << 2))); // ptr_shared_b[(row*2+1)*4]

            float4_add_mul(C_res[0], A_v1[(inner_k_count)&1], B_v1[(inner_k_count)&1].x);
            float4_add_mul(C_res[1], A_v2[(inner_k_count)&1], B_v1[(inner_k_count)&1].x);
            float4_add_mul(C_res[2], A_v1[(inner_k_count)&1], B_v1[(inner_k_count)&1].y);
            float4_add_mul(C_res[3], A_v2[(inner_k_count)&1], B_v1[(inner_k_count)&1].y);
            float4_add_mul(C_res[4], A_v1[(inner_k_count)&1], B_v1[(inner_k_count)&1].z);
            float4_add_mul(C_res[5], A_v2[(inner_k_count)&1], B_v1[(inner_k_count)&1].z);
            float4_add_mul(C_res[6], A_v1[(inner_k_count)&1], B_v1[(inner_k_count)&1].w);
            float4_add_mul(C_res[7], A_v2[(inner_k_count)&1], B_v1[(inner_k_count)&1].w);
            float4_add_mul(C_res[8], A_v1[(inner_k_count)&1], B_v2[(inner_k_count)&1].x);
            float4_add_mul(C_res[9], A_v2[(inner_k_count)&1], B_v2[(inner_k_count)&1].x);
            float4_add_mul(C_res[10], A_v1[(inner_k_count)&1], B_v2[(inner_k_count)&1].y);
            float4_add_mul(C_res[11], A_v2[(inner_k_count)&1], B_v2[(inner_k_count)&1].y);
            float4_add_mul(C_res[12], A_v1[(inner_k_count)&1], B_v2[(inner_k_count)&1].z);
            float4_add_mul(C_res[13], A_v2[(inner_k_count)&1], B_v2[(inner_k_count)&1].z);
            float4_add_mul(C_res[14], A_v1[(inner_k_count)&1], B_v2[(inner_k_count)&1].w);
            float4_add_mul(C_res[15], A_v2[(inner_k_count)&1], B_v2[(inner_k_count)&1].w);
        }

        if (k_count == K) {
            break;
        }

        // point to next shared buffer
        // and then load the next global value and write the global value
        // then sync
        auto _tmp_inc = k_count % K;
        auto _tmp_shared_buf_offset = (k_count & 1) << 10;
        ptr_shared_a = shared_a + _tmp_shared_buf_offset;
        ptr_shared_b = shared_b + _tmp_shared_buf_offset;
        //        ((float4 *)ptr_shared_a)[tx] = float4_val_ref(Matrix_addr(A_gpu, row_a, col_a, lda));
        //        _tmp_global_b = float4_val_ref(Matrix_addr(B_gpu, row_b, col_b, ldb));
        //        _tmp_shared_b_index = row_b + ((tx >> 3) << 5); // tx / 8 * 32 + tx % 8;
        //        ptr_shared_b[_tmp_shared_b_index] = _tmp_global_b.x;
        //        ptr_shared_b[_tmp_shared_b_index + 8] = _tmp_global_b.y;
        //        ptr_shared_b[_tmp_shared_b_index + 16] = _tmp_global_b.z;
        //        ptr_shared_b[_tmp_shared_b_index + 24] = _tmp_global_b.w;
        __syncthreads();
    }

    // load the value of origin C from global matrix
    float4_load(C_v[0], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_load(C_v[1], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_load(C_v[2], Matrix_addr(C_gpu, row_c + 1, col_c, ldc));
    float4_load(C_v[3], Matrix_addr(C_gpu, row_c + 1, col_c + 4, ldc));
    float4_load(C_v[4], Matrix_addr(C_gpu, row_c + 2, col_c, ldc));
    float4_load(C_v[5], Matrix_addr(C_gpu, row_c + 2, col_c + 4, ldc));
    float4_load(C_v[6], Matrix_addr(C_gpu, row_c + 3, col_c, ldc));
    float4_load(C_v[7], Matrix_addr(C_gpu, row_c + 3, col_c + 4, ldc));
    float4_load(C_v[8], Matrix_addr(C_gpu, row_c + 4, col_c, ldc));
    float4_load(C_v[9], Matrix_addr(C_gpu, row_c + 4, col_c + 4, ldc));
    float4_load(C_v[10], Matrix_addr(C_gpu, row_c + 5, col_c, ldc));
    float4_load(C_v[11], Matrix_addr(C_gpu, row_c + 5, col_c + 4, ldc));
    float4_load(C_v[12], Matrix_addr(C_gpu, row_c + 6, col_c, ldc));
    float4_load(C_v[13], Matrix_addr(C_gpu, row_c + 6, col_c + 4, ldc));
    float4_load(C_v[14], Matrix_addr(C_gpu, row_c + 7, col_c, ldc));
    float4_load(C_v[15], Matrix_addr(C_gpu, row_c + 7, col_c + 4, ldc));

    // add to res
    float4_add_mul_add_mull(C_res[0], alpha, C_res[0], beta, C_v[0]);
    float4_add_mul_add_mull(C_res[1], alpha, C_res[1], beta, C_v[1]);
    float4_add_mul_add_mull(C_res[2], alpha, C_res[2], beta, C_v[2]);
    float4_add_mul_add_mull(C_res[3], alpha, C_res[3], beta, C_v[3]);
    float4_add_mul_add_mull(C_res[4], alpha, C_res[4], beta, C_v[4]);
    float4_add_mul_add_mull(C_res[5], alpha, C_res[5], beta, C_v[5]);
    float4_add_mul_add_mull(C_res[6], alpha, C_res[6], beta, C_v[6]);
    float4_add_mul_add_mull(C_res[7], alpha, C_res[7], beta, C_v[7]);
    float4_add_mul_add_mull(C_res[8], alpha, C_res[8], beta, C_v[8]);
    float4_add_mul_add_mull(C_res[9], alpha, C_res[9], beta, C_v[9]);
    float4_add_mul_add_mull(C_res[10], alpha, C_res[10], beta, C_v[10]);
    float4_add_mul_add_mull(C_res[11], alpha, C_res[11], beta, C_v[11]);
    float4_add_mul_add_mull(C_res[12], alpha, C_res[12], beta, C_v[12]);
    float4_add_mul_add_mull(C_res[13], alpha, C_res[13], beta, C_v[13]);
    float4_add_mul_add_mull(C_res[14], alpha, C_res[14], beta, C_v[14]);
    float4_add_mul_add_mull(C_res[15], alpha, C_res[15], beta, C_v[15]);

    // store to global matrix C
    float4_store(C_res[0], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_store(C_res[1], Matrix_addr(C_gpu, row_c, col_c + 4, ldc));
    float4_store(C_res[2], Matrix_addr(C_gpu, row_c + 1, col_c, ldc));
    float4_store(C_res[3], Matrix_addr(C_gpu, row_c + 4, col_c + 4, ldc));
    float4_store(C_res[4], Matrix_addr(C_gpu, row_c + 2, col_c, ldc));
    float4_store(C_res[5], Matrix_addr(C_gpu, row_c + 2, col_c + 4, ldc));
    float4_store(C_res[6], Matrix_addr(C_gpu, row_c + 3, col_c, ldc));
    float4_store(C_res[7], Matrix_addr(C_gpu, row_c + 3, col_c + 4, ldc));
    float4_store(C_res[8], Matrix_addr(C_gpu, row_c + 4, col_c, ldc));
    float4_store(C_res[9], Matrix_addr(C_gpu, row_c + 4, col_c + 4, ldc));
    float4_store(C_res[10], Matrix_addr(C_gpu, row_c + 5, col_c, ldc));
    float4_store(C_res[11], Matrix_addr(C_gpu, row_c + 5, col_c + 4, ldc));
    float4_store(C_res[12], Matrix_addr(C_gpu, row_c + 6, col_c, ldc));
    float4_store(C_res[13], Matrix_addr(C_gpu, row_c + 6, col_c + 4, ldc));
    float4_store(C_res[14], Matrix_addr(C_gpu, row_c + 7, col_c, ldc));
    float4_store(C_res[15], Matrix_addr(C_gpu, row_c + 7, col_c + 4, ldc));
}

__host__ double gemm_k11(int TA, int TB, size_t M, size_t N, size_t K, float alpha, float *A_gpu, size_t lda,
                         float *B_gpu, size_t ldb, float beta, float *C_gpu, size_t ldc) {

    CUDA_CALL(cudaDeviceSynchronize());
    SET_TIME(t0)

    dim3 x;
    sgemm_nn<<<block_dim(), grid_dim(M, N)>>>(K, alpha, A_gpu, lda, B_gpu, ldb, beta, C_gpu, ldc);
    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t1)
    return GET_DURING(t1, t0);
}

