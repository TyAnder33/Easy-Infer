#include "easyinfer/gpt2.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef EASYINFER_ENABLE_CUDA
#include "easyinfer/device_buffer.hpp"
#include "easyinfer/kernels/context.hpp"
#include "easyinfer/kernels/ops.hpp"
#include "kernels/cuda_check.hpp"
#endif

namespace easyinfer {

GPT2::GPT2(ModelConfig config,
           const std::filesystem::path& model_directory,
           const std::vector<TensorMetadata>& metadata)
    : config_(std::move(config)) {
    if (config_.activation_function != "gelu_new") {
        throw std::runtime_error("unsupported GPT-2 activation '" +
                                 config_.activation_function + "'");
    }
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

std::vector<float> GPT2::forward(
    const std::vector<std::int32_t>& input_ids) {
#ifndef EASYINFER_ENABLE_CUDA
    static_cast<void>(input_ids);
    throw std::runtime_error("CUDA support is not enabled");
#else
    if (device_weights_.size() == 0) {
        throw std::runtime_error("device weights are not loaded");
    }
    if (input_ids.empty() ||
        input_ids.size() > static_cast<std::size_t>(config_.n_positions)) {
        throw std::runtime_error("input length is outside the model context");
    }
    for (const auto token_id : input_ids) {
        if (token_id < 0 || token_id >= config_.vocab_size) {
            throw std::runtime_error("input token ID is outside the vocabulary");
        }
    }

    const auto align = [](std::size_t count) {
        constexpr std::size_t alignment = 64;  // 256 bytes of FP32 values
        return (count + alignment - 1) & ~(alignment - 1);
    };

    const auto sequence = static_cast<std::int64_t>(input_ids.size());
    const auto hidden_size = config_.n_embd;
    const auto head_dim = config_.head_size();
    const auto hidden_count = static_cast<std::size_t>(sequence * hidden_size);
    const auto score_count = static_cast<std::size_t>(
        config_.n_head * sequence * sequence);
    const auto vocabulary = static_cast<std::size_t>(config_.vocab_size);
    const auto workspace_count =
        align(hidden_count) + align(hidden_count) +
        align(3 * hidden_count) + align(3 * hidden_count) +
        align(4 * hidden_count) + align(score_count) + align(vocabulary);

    DeviceBuffer device_input(input_ids.size() * sizeof(std::int32_t));
    DeviceBuffer workspace(workspace_count * sizeof(float));
    kernels::KernelContext context;

    kernels::detail::check_cuda(
        cudaMemcpy(device_input.data(),
                   input_ids.data(),
                   device_input.byte_size(),
                   cudaMemcpyHostToDevice),
        "copy input token IDs");

    auto* next = static_cast<float*>(workspace.data());
    const auto take = [&](std::size_t count) {
        float* result = next;
        next += align(count);
        return result;
    };

    float* hidden = take(hidden_count);
    float* normalized = take(hidden_count);
    float* packed = take(3 * hidden_count);
    float* heads = take(3 * hidden_count);
    float* mlp = take(4 * hidden_count);
    float* scores = take(score_count);
    float* logits = take(vocabulary);

    float* query = heads;
    float* key = query + hidden_count;
    float* value = key + hidden_count;
    float* attention = packed;
    float* merged = attention + hidden_count;

    const auto data = [](const DeviceTensorView& tensor) {
        return static_cast<const float*>(tensor.device_data);
    };
    const auto stream = context.stream();
    const auto epsilon = static_cast<float>(config_.layer_norm_epsilon);

    kernels::embedding_f32(
        stream,
        static_cast<const std::int32_t*>(device_input.data()),
        data(weights_.token_embedding),
        data(weights_.position_embedding),
        hidden,
        1,
        sequence,
        hidden_size,
        0);

    for (const auto& block : weights_.blocks) {
        kernels::layer_norm_f32(stream,
                                hidden,
                                data(block.ln_1.weight),
                                data(block.ln_1.bias),
                                normalized,
                                sequence,
                                hidden_size,
                                epsilon);
        kernels::linear_f32(context,
                            normalized,
                            data(block.attention.qkv.weight),
                            data(block.attention.qkv.bias),
                            packed,
                            sequence,
                            hidden_size,
                            3 * hidden_size,
                            kernels::WeightLayout::InputOutput);
        kernels::split_qkv_f32(stream,
                               packed,
                               query,
                               key,
                               value,
                               1,
                               sequence,
                               config_.n_head,
                               head_dim);
        kernels::causal_attention_f32(context,
                                      query,
                                      key,
                                      value,
                                      scores,
                                      attention,
                                      1,
                                      config_.n_head,
                                      sequence,
                                      sequence,
                                      head_dim,
                                      0);
        kernels::merge_heads_f32(stream,
                                 attention,
                                 merged,
                                 1,
                                 sequence,
                                 config_.n_head,
                                 head_dim);
        kernels::linear_f32(context,
                            merged,
                            data(block.attention.output.weight),
                            data(block.attention.output.bias),
                            normalized,
                            sequence,
                            hidden_size,
                            hidden_size,
                            kernels::WeightLayout::InputOutput);
        kernels::add_in_place_f32(
            stream, hidden, normalized, hidden_count);

        kernels::layer_norm_f32(stream,
                                hidden,
                                data(block.ln_2.weight),
                                data(block.ln_2.bias),
                                normalized,
                                sequence,
                                hidden_size,
                                epsilon);
        kernels::linear_f32(context,
                            normalized,
                            data(block.mlp.expansion.weight),
                            data(block.mlp.expansion.bias),
                            mlp,
                            sequence,
                            hidden_size,
                            4 * hidden_size,
                            kernels::WeightLayout::InputOutput);
        kernels::gelu_f32(stream, mlp, 4 * hidden_count);
        kernels::linear_f32(context,
                            mlp,
                            data(block.mlp.projection.weight),
                            data(block.mlp.projection.bias),
                            normalized,
                            sequence,
                            4 * hidden_size,
                            hidden_size,
                            kernels::WeightLayout::InputOutput);
        kernels::add_in_place_f32(
            stream, hidden, normalized, hidden_count);
    }

    kernels::layer_norm_f32(stream,
                            hidden,
                            data(weights_.final_layer_norm.weight),
                            data(weights_.final_layer_norm.bias),
                            normalized,
                            sequence,
                            hidden_size,
                            epsilon);
    kernels::linear_f32(
        context,
        normalized + (sequence - 1) * hidden_size,
        data(weights_.token_embedding),
        nullptr,
        logits,
        1,
        hidden_size,
        config_.vocab_size,
        kernels::WeightLayout::OutputInput);

    context.synchronize();
    std::vector<float> result(vocabulary);
    kernels::detail::check_cuda(cudaMemcpy(result.data(),
                                           logits,
                                           result.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost),
                                "copy output logits");
    return result;
#endif
}

}  // namespace easyinfer
