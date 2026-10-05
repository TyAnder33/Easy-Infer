#pragma once

#include <filesystem>
#include <string_view>
#include <vector>

#include "easyinfer/model.hpp"
#include "easyinfer/model_config.hpp"
#include "easyinfer/tensor.hpp"
#include "easyinfer/weight_store.hpp"

#ifdef EASYINFER_ENABLE_CUDA
#include "easyinfer/device_weight_store.hpp"
#endif

namespace easyinfer {

struct GPT2LinearWeights {
    DeviceTensorView weight;
    DeviceTensorView bias;
};

struct GPT2LayerNormWeights {
    DeviceTensorView weight;
    DeviceTensorView bias;
};

struct GPT2AttentionWeights {
    GPT2LinearWeights qkv;
    GPT2LinearWeights output;
};

struct GPT2MLPWeights {
    GPT2LinearWeights expansion;
    GPT2LinearWeights projection;
};

struct GPT2BlockWeights {
    GPT2LayerNormWeights ln_1;
    GPT2AttentionWeights attention;
    GPT2LayerNormWeights ln_2;
    GPT2MLPWeights mlp;
};

struct GPT2Weights {
    DeviceTensorView token_embedding;
    DeviceTensorView position_embedding;
    std::vector<GPT2BlockWeights> blocks;
    GPT2LayerNormWeights final_layer_norm;
};

class GPT2 final : public Model {
public:
    GPT2(ModelConfig config,
         const std::filesystem::path& model_directory,
         const std::vector<TensorMetadata>& metadata);

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::size_t weight_count() const override;
    void load_device_weights() override;
    void forward() override;

private:
    void validate_tensors() const;
#ifdef EASYINFER_ENABLE_CUDA
    void bind_device_weights();
#endif

    ModelConfig config_;
    WeightStore host_weights_;
#ifdef EASYINFER_ENABLE_CUDA
    DeviceWeightStore device_weights_;
#endif
    GPT2Weights weights_;
};

}  // namespace easyinfer
