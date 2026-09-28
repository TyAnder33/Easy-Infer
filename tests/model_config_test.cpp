#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "easyinfer/model_config.hpp"

namespace {

constexpr std::string_view kValidConfig = R"json(
{
  "activation_function": "gelu_new",
  "architectures": ["GPT2LMHeadModel"],
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 0.00001,
  "model_type": "gpt2",
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "vocab_size": 50257
}
)json";

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void test_valid_gpt2_config() {
    const auto config = easyinfer::parse_model_config(kValidConfig);
    expect(config.architecture == "GPT2LMHeadModel", "wrong architecture");
    expect(config.n_layer == 12, "wrong layer count");
    expect(config.n_embd == 768, "wrong embedding size");
    expect(config.head_size() == 64, "wrong attention head size");
}

void test_missing_field_is_rejected() {
    constexpr std::string_view json = R"json({"architectures": ["x"]})json";
    try {
        static_cast<void>(easyinfer::parse_model_config(json));
        throw std::runtime_error("missing field was accepted");
    } catch (const std::runtime_error& error) {
        expect(std::string_view(error.what()).find("model_type") !=
                   std::string_view::npos,
               "missing-field error lacks field name");
    }
}

void test_incompatible_attention_shape_is_rejected() {
    std::string json{kValidConfig};
    const auto position = json.find("\"n_head\": 12");
    json.replace(position, std::string_view("\"n_head\": 12").size(),
                 "\"n_head\": 7");

    try {
        static_cast<void>(easyinfer::parse_model_config(json));
        throw std::runtime_error("invalid attention shape was accepted");
    } catch (const std::runtime_error& error) {
        expect(std::string_view(error.what()).find("divisible") !=
                   std::string_view::npos,
               "attention-shape error is unclear");
    }
}

}  // namespace

int main() {
    try {
        test_valid_gpt2_config();
        test_missing_field_is_rejected();
        test_incompatible_attention_shape_is_rejected();
        std::cout << "model_config_test: all tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "model_config_test: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
