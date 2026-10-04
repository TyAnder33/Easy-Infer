#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "easyinfer/model.hpp"
#include "easyinfer/model_config.hpp"

namespace easyinfer {

[[nodiscard]] std::unique_ptr<Model> create_model(
    const ModelConfig& config,
    const std::filesystem::path& model_directory,
    const std::vector<TensorMetadata>& metadata);

}  // namespace easyinfer
