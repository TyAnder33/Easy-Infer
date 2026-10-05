#pragma once

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace easyinfer::kernels::detail {

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

inline void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

inline void check_last_kernel(const char* kernel_name) {
    check_cuda(cudaGetLastError(), kernel_name);
}

} // namespace easyinfer::kernels::detail
