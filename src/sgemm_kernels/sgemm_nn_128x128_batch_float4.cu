#include "sgemm_kernels.cuh"

/// assert(ldb%4==0 && ldc%4==0 && (lda%4==0 || K<8) )
__global__ __launch_bounds__(256) void sgemm_nn_128x128_batch_float4(unsigned int K, float alpha, const float *A_gpu,
                                                                     unsigned int lda, const float *B_gpu,
                                                                     unsigned int ldb, float beta, float *C_gpu,
                                                                     unsigned int ldc) {
    __shared__ float shared_a[256 * 4 * 2];
    __shared__ float shared_b[256 * 4 * 2];
    float *ptr_shared_a, *ptr_shared_b;
    float4 A_value1, A_value2, B_value1, B_value2, C_values[16], C_res[16];
    memset(C_res, 0, sizeof(C_res));

    auto _tmp_bx_mul_128 = blockIdx.x << 7;
    auto _tmp_by_mul_128 = blockIdx.y << 7;

    auto tx = threadIdx.x;
    // calculate the row and col
    auto _tmp_warp_id = tx >> 5;                                 // tx / 32
    auto _tmp_warp_row = _tmp_warp_id >> 2;                      // _tmp_warp_id / 4
    auto _tmp_warp_col = MOD4(_tmp_warp_id);                     // _tmp_warp_id % 4
    auto _tmp_lane_id = MOD32(tx);                               // tx % 32
    auto _tmp_row_in_warp = _tmp_lane_id >> 2;                   // _tmp_lane_id / 4
    auto _tmp_col_in_warp = MOD4(_tmp_lane_id);                  // _tmp_lane_id % 4
    auto row_c = (_tmp_warp_row << 6) + (_tmp_row_in_warp << 3); // _tmp_warp_row * 64 + _tmp_row_in_warp * 8;
    auto col_c = (_tmp_warp_col << 5) + (_tmp_col_in_warp << 3); // _tmp_warp_col * 32 + _tmp_col_in_warp * 8;

    // row and col for A, B to load from global matrix (load to shared memory)
    // change point to sub matrix
    auto row_a = tx >> 1;       // tx / 2
    auto col_a = MOD2(tx) << 2; // (tx % 2) * 4
    auto origin_a_gpu = A_gpu + _tmp_bx_mul_128 * lda;
    A_gpu = origin_a_gpu + IDX2F(row_a, col_a, lda);
    auto row_b = MOD8(tx);       // (tx % 8)
    auto col_b = (tx >> 3) << 2; // tx / 8 * 4
    B_gpu += _tmp_by_mul_128 + IDX2F(row_b, col_b, ldb);

    C_gpu += _tmp_bx_mul_128 * ldc + _tmp_by_mul_128;

    auto ldb_mul_8 = ldb << 3;
    auto batch_nums = K >> 3;
    auto batch_i = 1;
    if (batch_nums > 0) {
        while (true) {
            auto _tmp_global_a = float4_val_ref(A_gpu);
            auto _tmp_global_b = float4_val_ref(B_gpu);

            auto _tmp_shared_buf_offset = MOD2(batch_i) << 10;
            ptr_shared_a = shared_a + _tmp_shared_buf_offset;
            ptr_shared_b = shared_b + _tmp_shared_buf_offset;
            // every thread load the corresponding value from global matrix to shared memory and sync
            auto shared_a_index = ((tx >> 3) << 5) + (MOD2(tx) << 4) + (row_b >> 1); // tx/8*32 + tx%2*16 + tx%8/2
            ptr_shared_a[shared_a_index] = _tmp_global_a.x;
            ptr_shared_a[shared_a_index + 4] = _tmp_global_a.y;
            ptr_shared_a[shared_a_index + 8] = _tmp_global_a.z;
            ptr_shared_a[shared_a_index + 12] = _tmp_global_a.w;
            float4_ptr(ptr_shared_b)[tx] = _tmp_global_b;
            __syncthreads();

#pragma unroll
            for (auto _i = 0; _i < 8; ++_i) {
                // load from shared memory
                float4_load(A_value1, float4_ptr(ptr_shared_a) + (row_c << 1) + _i);
                float4_load(A_value2, float4_ptr(ptr_shared_a) + ((row_c << 1) + 8) + _i);
                float4_load(B_value1, float4_ptr(ptr_shared_b) + (col_c << 1) + _i);
                float4_load(B_value2, float4_ptr(ptr_shared_b) + ((col_c << 1) + 8) + _i);

                float4_add_mul(C_res[0], B_value1, A_value1.x);
                float4_add_mul(C_res[1], B_value2, A_value1.x);
                float4_add_mul(C_res[2], B_value1, A_value1.y);
                float4_add_mul(C_res[3], B_value2, A_value1.y);
                float4_add_mul(C_res[4], B_value1, A_value1.z);
                float4_add_mul(C_res[5], B_value2, A_value1.z);
                float4_add_mul(C_res[6], B_value1, A_value1.w);
                float4_add_mul(C_res[7], B_value2, A_value1.w);
                float4_add_mul(C_res[8], B_value1, A_value2.x);
                float4_add_mul(C_res[9], B_value2, A_value2.x);
                float4_add_mul(C_res[10], B_value1, A_value2.y);
                float4_add_mul(C_res[11], B_value2, A_value2.y);
                float4_add_mul(C_res[12], B_value1, A_value2.z);
                float4_add_mul(C_res[13], B_value2, A_value2.z);
                float4_add_mul(C_res[14], B_value1, A_value2.w);
                float4_add_mul(C_res[15], B_value2, A_value2.w);
            }

            if (batch_i == batch_nums) {
                break;
            }
            // point to next sub matrix
            A_gpu += 8;
            B_gpu += ldb_mul_8; // 8 * ldb
            ++batch_i;
        }
    }

    auto rest_nums = MOD8(K); // K%8
    // unlikely(rest_nums > 0)
    if (rest_nums > 0) {
        if (batch_nums > 0) {
            origin_a_gpu += (batch_nums << 3); // 8 * batch_nums
            B_gpu += ldb_mul_8;                // 8 * ldb
            auto _tmp_shared_buf_offset = MOD2(batch_i + 1) << 10;
            ptr_shared_a = shared_a + _tmp_shared_buf_offset;
            ptr_shared_b = shared_b + _tmp_shared_buf_offset;
        } else {
            ptr_shared_a = shared_a;
            ptr_shared_b = shared_b;
        }

        // load the corresponding value from global matrix to shared memory and sync
        if (row_b < rest_nums) { // if (tx%8 < rest_nums)
            A_gpu = origin_a_gpu + IDX2F(((tx >> 3) << 2), row_b, lda);
            // use A_value1 as the tmp value;
            A_value1.x = *A_gpu;
            A_value1.y = *(A_gpu + lda);
            A_value1.z = *(A_gpu + (lda << 1));
            A_value1.w = *(A_gpu + (lda << 1) + lda);
            float4_ptr(ptr_shared_a)[tx] = A_value1;
            float4_ptr(ptr_shared_b)[tx] = float4_val_ref(B_gpu);
        }
        __syncthreads();

        for (auto _i = 0; _i < rest_nums; ++_i) {
            // load from shared memory
            float4_load(A_value1, float4_ptr(ptr_shared_a) + (row_c << 1) + _i);
            float4_load(A_value2, float4_ptr(ptr_shared_a) + ((row_c << 1) + 8) + _i);
            float4_load(B_value1, float4_ptr(ptr_shared_b) + (col_c << 1) + _i);
            float4_load(B_value2, float4_ptr(ptr_shared_b) + ((col_c << 1) + 8) + _i);

            float4_add_mul(C_res[0], B_value1, A_value1.x);
            float4_add_mul(C_res[1], B_value2, A_value1.x);
            float4_add_mul(C_res[2], B_value1, A_value1.y);
            float4_add_mul(C_res[3], B_value2, A_value1.y);
            float4_add_mul(C_res[4], B_value1, A_value1.z);
            float4_add_mul(C_res[5], B_value2, A_value1.z);
            float4_add_mul(C_res[6], B_value1, A_value1.w);
            float4_add_mul(C_res[7], B_value2, A_value1.w);
            float4_add_mul(C_res[8], B_value1, A_value2.x);
            float4_add_mul(C_res[9], B_value2, A_value2.x);
            float4_add_mul(C_res[10], B_value1, A_value2.y);
            float4_add_mul(C_res[11], B_value2, A_value2.y);
            float4_add_mul(C_res[12], B_value1, A_value2.z);
            float4_add_mul(C_res[13], B_value2, A_value2.z);
            float4_add_mul(C_res[14], B_value1, A_value2.w);
            float4_add_mul(C_res[15], B_value2, A_value2.w);
        }
    }

    // load the value of origin C from global matrix
    float4_load(C_values[0], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_load(C_values[1], Matrix_addr(C_gpu, row_c, col_c + 4, ldc));
    float4_load(C_values[2], Matrix_addr(C_gpu, row_c + 1, col_c, ldc));
    float4_load(C_values[3], Matrix_addr(C_gpu, row_c + 1, col_c + 4, ldc));
    float4_load(C_values[4], Matrix_addr(C_gpu, row_c + 2, col_c, ldc));
    float4_load(C_values[5], Matrix_addr(C_gpu, row_c + 2, col_c + 4, ldc));
    float4_load(C_values[6], Matrix_addr(C_gpu, row_c + 3, col_c, ldc));
    float4_load(C_values[7], Matrix_addr(C_gpu, row_c + 3, col_c + 4, ldc));
    float4_load(C_values[8], Matrix_addr(C_gpu, row_c + 4, col_c, ldc));
    float4_load(C_values[9], Matrix_addr(C_gpu, row_c + 4, col_c + 4, ldc));
    float4_load(C_values[10], Matrix_addr(C_gpu, row_c + 5, col_c, ldc));
    float4_load(C_values[11], Matrix_addr(C_gpu, row_c + 5, col_c + 4, ldc));
    float4_load(C_values[12], Matrix_addr(C_gpu, row_c + 6, col_c, ldc));
    float4_load(C_values[13], Matrix_addr(C_gpu, row_c + 6, col_c + 4, ldc));
    float4_load(C_values[14], Matrix_addr(C_gpu, row_c + 7, col_c, ldc));
    float4_load(C_values[15], Matrix_addr(C_gpu, row_c + 7, col_c + 4, ldc));

    // add to res
    float4_add_mul_add_mull(C_res[0], alpha, C_res[0], beta, C_values[0]);
    float4_add_mul_add_mull(C_res[1], alpha, C_res[1], beta, C_values[1]);
    float4_add_mul_add_mull(C_res[2], alpha, C_res[2], beta, C_values[2]);
    float4_add_mul_add_mull(C_res[3], alpha, C_res[3], beta, C_values[3]);
    float4_add_mul_add_mull(C_res[4], alpha, C_res[4], beta, C_values[4]);
    float4_add_mul_add_mull(C_res[5], alpha, C_res[5], beta, C_values[5]);
    float4_add_mul_add_mull(C_res[6], alpha, C_res[6], beta, C_values[6]);
    float4_add_mul_add_mull(C_res[7], alpha, C_res[7], beta, C_values[7]);
    float4_add_mul_add_mull(C_res[8], alpha, C_res[8], beta, C_values[8]);
    float4_add_mul_add_mull(C_res[9], alpha, C_res[9], beta, C_values[9]);
    float4_add_mul_add_mull(C_res[10], alpha, C_res[10], beta, C_values[10]);
    float4_add_mul_add_mull(C_res[11], alpha, C_res[11], beta, C_values[11]);
    float4_add_mul_add_mull(C_res[12], alpha, C_res[12], beta, C_values[12]);
    float4_add_mul_add_mull(C_res[13], alpha, C_res[13], beta, C_values[13]);
    float4_add_mul_add_mull(C_res[14], alpha, C_res[14], beta, C_values[14]);
    float4_add_mul_add_mull(C_res[15], alpha, C_res[15], beta, C_values[15]);

    // store to global matrix C
    float4_store(C_res[0], Matrix_addr(C_gpu, row_c, col_c, ldc));
    float4_store(C_res[1], Matrix_addr(C_gpu, row_c, col_c + 4, ldc));
    float4_store(C_res[2], Matrix_addr(C_gpu, row_c + 1, col_c, ldc));
    float4_store(C_res[3], Matrix_addr(C_gpu, row_c + 1, col_c + 4, ldc));
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
