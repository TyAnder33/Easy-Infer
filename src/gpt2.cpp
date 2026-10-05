#include "easyinfer/gpt2.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace easyinfer {

GPT2::GPT2(ModelConfig config,
           const std::filesystem::path& model_directory,
           const std::vector<TensorMetadata>& metadata)
    : config_(std::move(config)) {
    host_weights_.load_weights(model_directory, metadata);
    validate_tensors();
}

std::string_view GPT2::name() const {
    return "GPT2";
}

std::size_t GPT2::weight_count() const {
    return host_weights_.size();
}

void GPT2::load_device_weights() {
#ifdef EASYINFER_ENABLE_CUDA
    device_weights_.load(host_weights_);
    bind_device_weights();
#else
    throw std::runtime_error("CUDA support is not enabled");
#endif
}

#ifdef EASYINFER_ENABLE_CUDA
void GPT2::bind_device_weights() {
    const auto linear = [this](const std::string& prefix) {
        return GPT2LinearWeights{
            .weight = device_weights_.at(prefix + ".weight"),
            .bias = device_weights_.at(prefix + ".bias"),
        };
    };
    const auto layer_norm = [this](const std::string& prefix) {
        return GPT2LayerNormWeights{
            .weight = device_weights_.at(prefix + ".weight"),
            .bias = device_weights_.at(prefix + ".bias"),
        };
    };

    weights_.token_embedding = device_weights_.at("wte.weight");
    weights_.position_embedding = device_weights_.at("wpe.weight");
    weights_.final_layer_norm = layer_norm("ln_f");
    weights_.blocks.resize(static_cast<std::size_t>(config_.n_layer));

    for (std::size_t layer = 0; layer < weights_.blocks.size(); ++layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";
        auto& block = weights_.blocks[layer];

        block.ln_1 = layer_norm(prefix + "ln_1");
        block.attention.qkv = linear(prefix + "attn.c_attn");
        block.attention.output = linear(prefix + "attn.c_proj");
        block.ln_2 = layer_norm(prefix + "ln_2");
        block.mlp.expansion = linear(prefix + "mlp.c_fc");
        block.mlp.projection = linear(prefix + "mlp.c_proj");
    }
}
#endif

void GPT2::validate_tensors() const {
    std::size_t expected_count = 0;

    const auto expect = [&](const std::string& name,
                            std::vector<std::int64_t> shape) {
        if (!host_weights_.contains(name)) {
            throw std::runtime_error("missing tensor '" + name + "'");
        }

        const auto& tensor = host_weights_.at(name);
        if (tensor.dtype != DType::FP32) {
            throw std::runtime_error("wrong dtype for tensor '" + name + "'");
        }
        if (tensor.shape != shape) {
            throw std::runtime_error("wrong shape for tensor '" + name + "'");
        }

        std::size_t element_count = 1;
        for (const auto dimension : shape) {
            element_count *= static_cast<std::size_t>(dimension);
        }
        if (tensor.byte_size != element_count * sizeof(float)) {
            throw std::runtime_error("wrong byte size for tensor '" + name +
                                     "'");
        }

        ++expected_count;
    };

    const auto hidden = config_.n_embd;
    const auto mlp_hidden = 4 * hidden;

    expect("wte.weight", {config_.vocab_size, hidden});
    expect("wpe.weight", {config_.n_positions, hidden});
    expect("ln_f.weight", {hidden});
    expect("ln_f.bias", {hidden});

    for (std::int64_t layer = 0; layer < config_.n_layer; ++layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";

        expect(prefix + "attn.bias",
               {1, 1, config_.n_positions, config_.n_positions});
        expect(prefix + "attn.c_attn.weight", {hidden, 3 * hidden});
        expect(prefix + "attn.c_attn.bias", {3 * hidden});
        expect(prefix + "attn.c_proj.weight", {hidden, hidden});
        expect(prefix + "attn.c_proj.bias", {hidden});
        expect(prefix + "ln_1.weight", {hidden});
        expect(prefix + "ln_1.bias", {hidden});
        expect(prefix + "ln_2.weight", {hidden});
        expect(prefix + "ln_2.bias", {hidden});
        expect(prefix + "mlp.c_fc.weight", {hidden, mlp_hidden});
        expect(prefix + "mlp.c_fc.bias", {mlp_hidden});
        expect(prefix + "mlp.c_proj.weight", {mlp_hidden, hidden});
        expect(prefix + "mlp.c_proj.bias", {hidden});
    }

    if (host_weights_.size() != expected_count) {
        throw std::runtime_error("checkpoint contains unexpected tensors");
    }
}

void GPT2::forward() {
    throw std::runtime_error("GPT-2 forward pass is not implemented");
}

}  // namespace easyinfer
