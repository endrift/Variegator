set(CMAKE_INCLUDE_CURRENT_DIR ON)
include(FindCUDA)

set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} --generate-line-info")
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G")
cuda_add_library(v_cuda cuda.cu STATIC)
cuda_add_executable(cuvary main.cu)
target_link_libraries(cuvary v_cuda c)
