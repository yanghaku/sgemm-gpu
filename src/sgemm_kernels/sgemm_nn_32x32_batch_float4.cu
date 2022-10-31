#include "sgemm_kernels.cuh"

/// assert(ldb%4==0 && ldc%4==0 && (lda%4==0 || K<8) )
__global__ __launch_bounds__(64) void sgemm_nn_32x32_batch_float4(unsigned int K, float alpha, const float *A_gpu,
                                                                  unsigned int lda, const float *B_gpu,
                                                                  unsigned int ldb, float beta, float *C_gpu,
                                                                  unsigned int ldc) {
    __shared__ float shared_a[64 * 4 * 2];
    __shared__ float shared_b[64 * 4 * 2];
    float *ptr_shared_a, *ptr_shared_b;
    float4 A_value, B_value, C_values[4], C_res[4];
    memset(C_res, 0, sizeof(C_res));

    auto _tmp_bx_mul_32 = blockIdx.x << 5;
    auto _tmp_by_mul_32 = blockIdx.y << 5;
    auto tx = threadIdx.x;

    // calculate the row and col
    auto row_c = (tx >> 3) << 2;
    auto col_c = MOD8(tx) << 2;

    // row and col for A, B to load from global matrix (load to shared memory)
    // change point to sub matrix
    auto row_a = ((tx >> 3) << 2) + MOD4(tx); // tx / 8 * 4 + tx%4
    auto col_a = (MOD8(tx) >> 2) << 2;        // tx % 8 / 4 * 4
    auto origin_a_gpu = A_gpu + _tmp_bx_mul_32 * lda;
    A_gpu = origin_a_gpu + IDX2F(row_a, col_a, lda);
    auto row_b = MOD8(tx);       // tx % 8
    auto col_b = (tx >> 3) << 2; // tx / 8 * 4
    B_gpu += _tmp_by_mul_32 + IDX2F(row_b, col_b, ldb);
    C_gpu += _tmp_bx_mul_32 * ldc + _tmp_by_mul_32;

    auto ldb_mul_8 = ldb << 3;
    auto batch_nums = K >> 3; // batch = 8
    auto batch_i = 1;
    if (batch_nums > 0) {
        while (true) {
            auto _tmp_global_a = float4_val_ref(A_gpu);
            auto _tmp_global_b = float4_val_ref(B_gpu);

            auto _tmp_shared_buf_offset = MOD2(batch_i) << 8;
            ptr_shared_a = shared_a + _tmp_shared_buf_offset;
            ptr_shared_b = shared_b + _tmp_shared_buf_offset;
            // every thread load the corresponding value from global matrix to shared memory and sync
            auto shared_a_index = ((tx >> 2) << 4) + MOD4(tx); // tx/4*16 + tx%4
            ptr_shared_a[shared_a_index] = _tmp_global_a.x;
            ptr_shared_a[shared_a_index + 4] = _tmp_global_a.y;
            ptr_shared_a[shared_a_index + 8] = _tmp_global_a.z;
            ptr_shared_a[shared_a_index + 12] = _tmp_global_a.w;
            float4_ptr(ptr_shared_b)[tx] = _tmp_global_b;
            __syncthreads();

            ptr_shared_a += (row_c << 3);
            ptr_shared_b += (col_c << 3);
            auto _i = 1;
#pragma unroll
            while (true) {
                float4_load(A_value, float4_ptr(ptr_shared_a));
                float4_load(B_value, float4_ptr(ptr_shared_b));
                float4_add_mul(C_res[0], B_value, A_value.x);
                float4_add_mul(C_res[1], B_value, A_value.y);
                float4_add_mul(C_res[2], B_value, A_value.z);
                float4_add_mul(C_res[3], B_value, A_value.w);
                if (_i == 8) {
                    break;
                } else {
                    ++_i;
                    ptr_shared_a += 4;
                    ptr_shared_b += 4;
                }
            }

            if (batch_i == batch_nums) {
                break;
            }
            ++batch_i;
            // point to next sub matrix
            A_gpu += 8;
            B_gpu += ldb_mul_8; // 8 * ldb
        }
    }

    auto rest_nums = MOD8(K); // K%8
    if (rest_nums > 0) {      // unlikely(rest_nums > 0)
        if (batch_nums > 0) {
            auto _tmp_shared_buf_offset = MOD2(batch_i + 1) << 8;
            ptr_shared_a = shared_a + _tmp_shared_buf_offset;
            ptr_shared_b = shared_b + _tmp_shared_buf_offset;
            B_gpu += ldb_mul_8;                // 8 * ldb
            origin_a_gpu += (batch_nums << 3); // 8 * batch_nums
        } else {
            ptr_shared_a = shared_a;
            ptr_shared_b = shared_b;
        }

        // load the corresponding value from global matrix to shared memory and sync
        if (row_b < rest_nums) { // if (tx%8 < rest_nums)
            A_gpu = origin_a_gpu + IDX2F(col_b, row_b, lda);
            // use A_value as the tmp value;
            A_value.x = *A_gpu;
            A_value.y = *(A_gpu + lda);
            A_value.z = *(A_gpu + (lda << 1));
            A_value.w = *(A_gpu + (lda << 1) + lda);
            float4_ptr(ptr_shared_a)[tx] = A_value;
            float4_ptr(ptr_shared_b)[tx] = float4_val_ref(B_gpu);
        }
        __syncthreads();

        for (auto _i = 0; _i < rest_nums; ++_i) {
            // load from shared memory
            float4_load(A_value, float4_ptr(ptr_shared_a) + (row_c << 1) + _i);
            float4_load(B_value, float4_ptr(ptr_shared_b) + (col_c << 1) + _i);
            float4_add_mul(C_res[0], B_value, A_value.x);
            float4_add_mul(C_res[1], B_value, A_value.y);
            float4_add_mul(C_res[2], B_value, A_value.z);
            float4_add_mul(C_res[3], B_value, A_value.w);
        }
    }

    // load the value of origin C from global matrix
    float4_load(C_values[0], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_load(C_values[1], Matrix_addr(C_gpu, row_c + 1, col_c, ldc));
    float4_load(C_values[2], Matrix_addr(C_gpu, row_c + 2, col_c, ldc));
    float4_load(C_values[3], Matrix_addr(C_gpu, row_c + 3, col_c, ldc));

    // add to res
    float4_add_mul_add_mull(C_res[0], alpha, C_res[0], beta, C_values[0]);
    float4_add_mul_add_mull(C_res[1], alpha, C_res[1], beta, C_values[1]);
    float4_add_mul_add_mull(C_res[2], alpha, C_res[2], beta, C_values[2]);
    float4_add_mul_add_mull(C_res[3], alpha, C_res[3], beta, C_values[3]);

    // store to global matrix C
    float4_store(C_res[0], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_store(C_res[1], Matrix_addr(C_gpu, row_c + 1, col_c, ldc));
    float4_store(C_res[2], Matrix_addr(C_gpu, row_c + 2, col_c, ldc));
    float4_store(C_res[3], Matrix_addr(C_gpu, row_c + 3, col_c, ldc));
}
