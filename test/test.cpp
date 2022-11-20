#include "cuda_helper.h"
#include "sgemm_gpu.h"
#include "time_helper.h"
#include <cstring>
#include <cublas_v2.h>

extern "C" double gemm_cpu(int TA, int TB, unsigned int M, unsigned int N, unsigned int K, float ALPHA, const float *A,
                           unsigned int lda, const float *B, unsigned int ldb, float BETA, float *C, unsigned int ldc);

double call_gemm_cublas(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, float *A_gpu, size_t lda,
                        float *B_gpu, size_t ldb, float BETA, float *C_gpu, size_t ldc) {
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));
    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t0)
    CUBLAS_CALL(cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA,
                            B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc));
    CUDA_CALL(cudaDeviceSynchronize());
    SET_TIME(t1)

    CUBLAS_CALL(cublasDestroy(handle));
    return GET_DURING(t1, t0);
}

double call_gemm_gpu(int TA, int TB, size_t M, size_t N, size_t K, float ALPHA, float *A_gpu, size_t lda, float *B_gpu,
                     size_t ldb, float BETA, float *C_gpu, size_t ldc) {

    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t0)
    sgemm_gpu(TA, TB, (int)M, (int)N, (int)K, ALPHA, A_gpu, (int)lda, B_gpu, (int)ldb, BETA, C_gpu, (int)ldc);
    CUDA_CALL(cudaDeviceSynchronize());

    SET_TIME(t1)
    return GET_DURING(t1, t0);
}

constexpr auto cpu_check_eps = 1e-2;
constexpr auto gpu_check_eps = 1e-4;

#define print_args()                                                                                                   \
    do {                                                                                                               \
        std::cout << "TA=" << TA << ",TB=" << TB << std::endl;                                                         \
        std::cout << "m=" << m << ",k=" << k << ",n=" << n << std::endl;                                               \
        std::cout << "lda=" << lda << ",ldb=" << ldb << ",ldc=" << ldc << std::endl;                                   \
        std::cout << "a_count=" << a_count << ",b_count=" << b_count << ",c_count=" << c_count << std::endl;           \
        std::cout << "FLOP = " << 2 * m * k * n << std::endl;                                                          \
    } while (0)

#define report_error(msg, out_var, correct_out_var, err_id)                                                            \
    do {                                                                                                               \
        std::cout.precision(17);                                                                                       \
        if (!verbose) {                                                                                                \
            print_args();                                                                                              \
        }                                                                                                              \
        std::cerr << "Test Error: expect " msg "[" << err_id << "] = " << std::fixed << correct_out_var[err_id]        \
                  << " but get " << std::fixed << out_var[err_id] << std::endl;                                        \
        cudaDeviceReset();                                                                                             \
        std::exit(-1);                                                                                                 \
    } while (0)

#define report_time(msg, time)                                                                                         \
    do {                                                                                                               \
        std::cout.precision(17);                                                                                       \
        auto flop = 2 * m * k * n;                                                                                     \
        auto gflops = ((double)(flop)) / time * 1e-6;                                                                  \
        std::cerr << "Time " msg " = \t" << time << " ms; (m,k,n)=(" << m << "," << k << "," << n                      \
                  << "); GFLOPS=" << std::fixed << gflops << std::endl;                                                \
    } while (0)

void do_test(int TA, int TB, size_t m, size_t k, size_t n, size_t lda, size_t ldb, size_t ldc, size_t a_count,
             size_t b_count, size_t c_count, bool output_time = false, bool verbose = false, bool cpu = false) {
    if (verbose) {
        output_time = true;
        print_args();
    }

    SET_TIME(time_0)
    std::default_random_engine g(TO_SEED(time_0));
    std::uniform_real_distribution<float> d(-1, 1);
    auto alpha = d(g);
    auto beta = d(g);

    auto a = new float[a_count];
    auto b = new float[b_count];
    auto out_c = new float[c_count];
    randomize_matrix(a, a_count);
    randomize_matrix(b, b_count);
    auto gpu_a = create_cuda_matrix(a, a_count);
    auto gpu_b = create_cuda_matrix(b, b_count);
    auto gpu_c = create_cuda_matrix(nullptr, c_count);

    // compute with cublas
    auto cublas_out_c = new float[c_count];
    auto time_cublas = call_gemm_cublas(TA, TB, m, n, k, alpha, gpu_a, lda, gpu_b, ldb, beta, gpu_c, ldc);
    if (output_time) {
        report_time("cublas", time_cublas);
    }
    CUDA_CALL(cudaMemcpy(cublas_out_c, gpu_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    // compute with kernel
    CUDA_CALL(cudaMemset(gpu_c, 0, c_count * sizeof(float)));
    CUDA_CALL(cudaDeviceSynchronize());
    auto time_my_kernel = call_gemm_gpu(TA, TB, m, n, k, alpha, gpu_a, lda, gpu_b, ldb, beta, gpu_c, ldc);
    if (output_time) {
        report_time("my", time_my_kernel);
    }
    CUDA_CALL(cudaMemcpy(out_c, gpu_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaDeviceSynchronize());

    auto my_kernel_err_id = check_matrix(cublas_out_c, out_c, c_count, gpu_check_eps);
    if (my_kernel_err_id != c_count) {
        report_error("my", out_c, cublas_out_c, my_kernel_err_id);
    }

    // compute with cpu
    if (cpu) {
        memset(out_c, 0, sizeof(float) * c_count);
        auto time_cpu = gemm_cpu(TA, TB, m, n, k, alpha, a, lda, b, ldb, beta, out_c, ldc);
        if (output_time) {
            report_time("cpu", time_cpu);
        }
        auto cpu_err_id = check_matrix(cublas_out_c, out_c, c_count, cpu_check_eps);
        if (cpu_err_id != c_count) {
            report_error("cpu", out_c, cublas_out_c, cpu_err_id);
        }
    }

    // free all
    CUDA_CALL(cudaFree(gpu_c));
    CUDA_CALL(cudaFree(gpu_b));
    CUDA_CALL(cudaFree(gpu_a));
    delete[] cublas_out_c;
    delete[] out_c;
    delete[] b;
    delete[] a;
    if (verbose) {
        std::cerr << "---------------------OK-----------------" << std::endl << std::endl;
    }
}

void test_nn(bool output_time = true, bool verbose = false, bool with_cpu = false) {
#define test_simple_nn(m, k, n)                                                                                        \
    do_test(0, 0, m, k, n, k, n, n, (m) * (k), (k) * (n), (m) * (n), output_time, verbose, with_cpu)

    test_simple_nn(128 + 32, 8, 128);
    test_simple_nn(32, 1, 32);
    test_simple_nn(512, 5, 64);
    test_simple_nn(1024, 8, 32);
    test_simple_nn(64, 68, 512);
    test_simple_nn(1024, 1, 512);
    test_simple_nn(256, 5, 512);
    test_simple_nn(128, 8, 128);
    test_simple_nn(256 + 32, 12, 512 + 32);
    test_simple_nn(256, 1028, 128);
    test_simple_nn(128, 128, 256 + 64 + 32);
    test_simple_nn(1024 + 64, 1024, 1024);
    test_simple_nn(2048, 2048, 2048);
    test_simple_nn(4096, 4096, 4096);

#define test_custom(m, k, n, lda, ldb, ldc)                                                                            \
    do_test(0, 0, m, k, n, lda, ldb, ldc, (m) * (lda), (k) * (ldb), (m) * (ldc), output_time, verbose, with_cpu)

    test_custom(16, 27, 173056, 27, 173056, 173056);
    test_custom(32, 144, 43264, 144, 43264, 43264);
    test_custom(64, 288, 10816, 288, 10816, 10816);
    test_custom(128, 576, 2704, 576, 2704, 2704);
    test_custom(256, 1152, 676, 1152, 676, 676);
    test_custom(512, 2304, 169, 2304, 169, 169);
    test_custom(1024, 4608, 169, 4608, 169, 169);
    test_custom(256, 1024, 169, 1024, 169, 169);
    test_custom(512, 2304, 169, 2304, 169, 169);
    test_custom(255, 512, 169, 512, 169, 169);
    test_custom(128, 256, 169, 256, 169, 169);
    test_custom(256, 3456, 676, 3456, 676, 676);
    test_custom(255, 256, 676, 256, 676, 676);

    //    test_simple_nn(64, 2916, 363);
    //    test_simple_nn(192, 729, 1600);
    //    test_simple_nn(384, 196, 1728);
    //    test_simple_nn(256, 196, 3456);
    //    test_simple_nn(256, 196, 2304);
    //    test_simple_nn(128, 4096, 12544);
    //    test_simple_nn(128, 4096, 4096);
    //    test_simple_nn(64, 75, 12544);
    //    test_simple_nn(64, 75, 12544);
    //    test_simple_nn(64, 75, 12544);
    //    test_simple_nn(64, 576, 12544);
    //    test_simple_nn(256, 2304, 784);
    //    do_test(1, 1, 2304, 256, 784);
    //    test_simple_nn(512, 4608, 196);
    //    do_test(1, 1, 4608, 512, 196);
}

static inline void print_helper() {
    std::cout << "-t: just output time" << std::endl;
    std::cout << "-v: output time and other verbose message" << std::endl;
    std::cout << "-i: interactive test" << std::endl;
    std::cout << "-c: test run with cpu" << std::endl;
    std::cout << "-h: print help message" << std::endl;
}

int main(int argc, char *argv[]) {
    bool verbose = false;
    bool output_time = false;
    bool interactive = false;
    bool with_cpu = false;
    for (auto i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
        } else if (!strcmp(argv[i], "-t")) {
            output_time = true;
        } else if (!strcmp(argv[i], "-i")) {
            interactive = true;
        } else if (!strcmp(argv[i], "-h")) {
            print_helper();
            std::exit(0);
        } else if (!strcmp(argv[i], "-c")) {
            with_cpu = true;
        } else {
            std::cerr << "unknown argument: " << argv[i] << std::endl;
            print_helper();
        }
    }

    CUDA_CALL(cudaSetDevice(0));

    if (interactive) {
        int TA, TB;
        size_t m, k, n, lda, ldb, ldc, a_count, b_count, c_count;
        std::cout << "input TA TB:" << std::endl;
        std::cin >> TA >> TB;
        std::cout << "input m k n:" << std::endl;
        std::cin >> m >> k >> n;
        std::cout << "input lda ldb ldc:" << std::endl;
        std::cin >> lda >> ldb >> ldc;
        std::cout << "input a_count b_count c_count:" << std::endl;
        std::cin >> a_count >> b_count >> c_count;
        do_test(TA, TB, m, k, n, lda, ldb, ldc, a_count, b_count, c_count, output_time, verbose, with_cpu);
    } else {
        test_nn(output_time, verbose, with_cpu);
    }

    CUDA_CALL(cudaDeviceReset());
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
