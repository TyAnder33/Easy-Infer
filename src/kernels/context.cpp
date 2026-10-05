#include "easyinfer/kernels/context.hpp"

#include "cuda_check.hpp"

namespace easyinfer::kernels {

KernelContext::KernelContext(std::size_t workspace_bytes)
    : workspace_bytes_(workspace_bytes) {
    detail::check_cuda(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "create CUDA stream");

    try {
        detail::check_cublas(cublasLtCreate(&cublas_lt_),
                             "create cuBLASLt handle");
        if (workspace_bytes_ != 0) {
            detail::check_cuda(cudaMalloc(&workspace_, workspace_bytes_),
                               "allocate cuBLASLt workspace");
        }
    } catch (...) {
        if (cublas_lt_ != nullptr) {
            cublasLtDestroy(cublas_lt_);
        }
        cudaStreamDestroy(stream_);
        throw;
    }
}

KernelContext::~KernelContext() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
    }
    if (workspace_ != nullptr) {
        cudaFree(workspace_);
    }
    if (cublas_lt_ != nullptr) {
        cublasLtDestroy(cublas_lt_);
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void KernelContext::synchronize() const {
    detail::check_cuda(cudaStreamSynchronize(stream_),
                       "synchronize CUDA stream");
}

} // namespace easyinfer::kernels
