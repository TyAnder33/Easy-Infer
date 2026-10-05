#include "easyinfer/device_weight_store.hpp"

#include "cuda_check.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace easyinfer {
namespace {

constexpr std::size_t kAlignment = 256;

std::size_t align_up(std::size_t byte_size) {
    return (byte_size + kAlignment - 1) & ~(kAlignment - 1);
}

}  // namespace

DeviceWeightStore::~DeviceWeightStore() {
    if (allocation_ != nullptr) {
        cudaFree(allocation_);
    }
}

void DeviceWeightStore::load(const WeightStore& host_weights) {
    if (allocation_ != nullptr) {
        throw std::runtime_error("device weights are already loaded");
    }
    const auto& host_tensors = host_weights.tensors();
    if (host_tensors.empty()) {
        throw std::runtime_error("cannot upload an empty weight store");
    }

    std::size_t allocation_size = 0;
    for (const auto& entry : host_tensors) {
        allocation_size += align_up(entry.second.byte_size);
    }

    void* allocation = nullptr;
    kernels::detail::check_cuda(cudaMalloc(&allocation, allocation_size),
                                "allocate device weights");

    std::unordered_map<std::string, DeviceTensorView> tensors;
    tensors.reserve(host_tensors.size());
    auto* next = static_cast<std::byte*>(allocation);

    try {
        for (const auto& [name, tensor] : host_tensors) {
            const std::string operation = "copy tensor '" + name + "'";
            kernels::detail::check_cuda(
                cudaMemcpy(next,
                           tensor.host_data,
                           tensor.byte_size,
                           cudaMemcpyHostToDevice),
                operation.c_str());

            tensors.emplace(name,
                            DeviceTensorView{
                                .device_data = next,
                                .dtype = tensor.dtype,
                                .shape = tensor.shape,
                                .byte_size = tensor.byte_size,
                            });
            next += align_up(tensor.byte_size);
        }
    } catch (...) {
        cudaFree(allocation);
        throw;
    }

    tensors_.swap(tensors);
    allocation_ = allocation;
}

const DeviceTensorView& DeviceWeightStore::at(const std::string& name) const {
    return tensors_.at(name);
}

}  // namespace easyinfer
