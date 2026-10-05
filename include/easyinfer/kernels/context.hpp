#pragma once

#include <cstddef>

#include <cublasLt.h>
#include <cuda_runtime_api.h>

namespace easyinfer::kernels {

class KernelContext final {
public:
    explicit KernelContext(std::size_t workspace_bytes = 32 * 1024 * 1024);
    ~KernelContext();

    KernelContext(const KernelContext&) = delete;
    KernelContext& operator=(const KernelContext&) = delete;
    KernelContext(KernelContext&&) = delete;
    KernelContext& operator=(KernelContext&&) = delete;

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    [[nodiscard]] cublasLtHandle_t cublas_lt() const noexcept { return cublas_lt_; }
    [[nodiscard]] void* workspace() const noexcept { return workspace_; }
    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return workspace_bytes_;
    }

    void synchronize() const;

private:
    cudaStream_t stream_{};
    cublasLtHandle_t cublas_lt_{};
    void* workspace_{};
    std::size_t workspace_bytes_{};
};

} // namespace easyinfer::kernels
