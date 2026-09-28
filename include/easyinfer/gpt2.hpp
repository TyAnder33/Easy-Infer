#pragma once

#include <vector>

#include "easyinfer/model.hpp"
#include "easyinfer/tensor.hpp"

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
    GPT2() = default;

    void forward() override;

private:
    GPT2Weights weights_;
};

}  // namespace easyinfer
