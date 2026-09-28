#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace easyinfer {

// The subset of Hugging Face's GPT-2 configuration that affects model shape
// or inference behavior. Keep the original Hugging Face field names so it is
// easy to compare this object with config.json.
struct ModelConfig {
    std::string architecture;
    std::string model_type;
    std::string activation_function;

    std::int64_t vocab_size{};
    std::int64_t n_positions{};
    std::int64_t n_embd{};
    std::int64_t n_layer{};
    std::int64_t n_head{};
    std::int64_t bos_token_id{};
    std::int64_t eos_token_id{};

    double layer_norm_epsilon{};
    double initializer_range{};

    [[nodiscard]] std::int64_t head_size() const;
};

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t data_begin{};
    std::uint64_t data_end{};
};

// Parsing text separately from file I/O makes malformed configurations easy
// to unit test without creating temporary files.
[[nodiscard]] ModelConfig parse_model_config(std::string_view json_text);

[[nodiscard]] ModelConfig load_model_config(
    const std::filesystem::path& model_directory);

[[nodiscard]] std::vector<TensorMetadata> load_safetensors_metadata(
    const std::filesystem::path& model_directory);

}  // namespace easyinfer
