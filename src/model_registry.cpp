#include "easyinfer/model_registry.hpp"

#include <memory>
#include <stdexcept>

#include "easyinfer/gpt2.hpp"

namespace easyinfer {

std::unique_ptr<Model> create_model(
    const ModelConfig& config,
    const std::filesystem::path& model_directory,
    const std::vector<TensorMetadata>& metadata) {
    if (config.architecture == "GPT2LMHeadModel") {
        return std::make_unique<GPT2>(config, model_directory, metadata);
    }

    throw std::runtime_error("unsupported architecture '" +
                             config.architecture + "'");
}

}  // namespace easyinfer
