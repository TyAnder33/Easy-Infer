#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "easyinfer/model_config.hpp"
#include "easyinfer/tensor.hpp"

namespace easyinfer {

class WeightStore {
public:
    WeightStore() = default;
    ~WeightStore();

    WeightStore(const WeightStore&) = delete;
    WeightStore& operator=(const WeightStore&) = delete;

    void load_weights(
        const std::filesystem::path& model_directory,
        const std::vector<TensorMetadata>& metadata);

    [[nodiscard]] bool contains(const std::string& name) const;
    [[nodiscard]] const TensorView& at(const std::string& name) const;
    [[nodiscard]] std::size_t size() const;

private:
    void* mapped_file_{nullptr};
    std::size_t mapped_file_size_{};
    std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace easyinfer
