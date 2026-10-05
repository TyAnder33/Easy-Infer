#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

#include "easyinfer/tensor.hpp"
#include "easyinfer/weight_store.hpp"

namespace easyinfer {

class DeviceWeightStore {
public:
    DeviceWeightStore() = default;
    ~DeviceWeightStore();

    DeviceWeightStore(const DeviceWeightStore&) = delete;
    DeviceWeightStore& operator=(const DeviceWeightStore&) = delete;

    void load(const WeightStore& host_weights);

    [[nodiscard]] const DeviceTensorView& at(const std::string& name) const;
    [[nodiscard]] std::size_t size() const noexcept { return tensors_.size(); }

private:
    void* allocation_{nullptr};
    std::unordered_map<std::string, DeviceTensorView> tensors_;
};

}  // namespace easyinfer
