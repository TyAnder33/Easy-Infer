#include "easyinfer/device_buffer.hpp"

#include "cuda_check.hpp"

#include <utility>

namespace easyinfer {

DeviceBuffer::DeviceBuffer(std::size_t byte_size) : byte_size_(byte_size) {
    if (byte_size_ != 0) {
        kernels::detail::check_cuda(cudaMalloc(&data_, byte_size_),
                                    "allocate device buffer");
    }
}

DeviceBuffer::~DeviceBuffer() {
    if (data_ != nullptr) {
        cudaFree(data_);
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      byte_size_(std::exchange(other.byte_size_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = std::exchange(other.data_, nullptr);
        byte_size_ = std::exchange(other.byte_size_, 0);
    }
    return *this;
}

}  // namespace easyinfer
