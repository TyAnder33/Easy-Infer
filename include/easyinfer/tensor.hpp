#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace easyinfer {

enum class DType {
    FP32,
};

struct TensorView {
    const void* host_data{nullptr};
    DType dtype{DType::FP32};
    std::vector<std::int64_t> shape;
    std::size_t byte_size{};
};

struct DeviceTensorView {
    const void* device_data{nullptr};
    DType dtype{DType::FP32};
    std::vector<std::int64_t> shape;
    std::size_t byte_size{};
};

}  // namespace easyinfer
