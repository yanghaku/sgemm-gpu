cmake_minimum_required(VERSION 3.20)

project(sgemm_gpu)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS} include)
enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions -fPIC -fPIE -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -dlto -rdc true -noeh -m64 -dopt=on -ccbin ${CMAKE_CXX_COMPILER} -allow-unsupported-compiler -Xcompiler '${CMAKE_CXX_FLAGS}'")

file(GLOB kernels src/sgemm_kernels/*.cu)
add_library(sgemm_gpu STATIC src/sgemm_gpu.cu ${kernels})

add_executable(test test/test.cpp test/sgemm_cpu.c)
target_link_libraries(test PRIVATE CUDA::cublas sgemm_gpu)
