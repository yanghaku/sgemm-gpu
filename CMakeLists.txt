cmake_minimum_required(VERSION 3.23)

project(cublas_gemm)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions -O3 -s -flto")
#add_definitions(-DRUN_CPU)

add_executable(test test.cpp kernel11.cu)
target_link_libraries(test PRIVATE CUDA::cublas)
