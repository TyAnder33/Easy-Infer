#include "easyinfer/model_config.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace easyinfer {
namespace {

using Json = nlohmann::json;

template <typename T>
T required(const Json& json, const char* field) {
    if (!json.contains(field)) {
        throw std::runtime_error("config.json is missing required field '" +
                                 std::string(field) + "'");
    }

    try {
        return json.at(field).get<T>();
    } catch (const Json::exception& error) {
        throw std::runtime_error("invalid config.json field '" +
                                 std::string(field) + "': " + error.what());
    }
}

void require_positive(std::int64_t value, const char* field) {
    if (value <= 0) {
        throw std::runtime_error("config.json field '" + std::string(field) +
                                 "' must be positive");
    }
}

void validate(const ModelConfig& config) {
    if (config.architecture.empty()) {
        throw std::runtime_error(
            "config.json field 'architectures[0]' must not be empty");
    }
    if (config.model_type != "gpt2") {
        throw std::runtime_error("unsupported model_type '" + config.model_type +
                                 "' (expected 'gpt2')");
    }

    require_positive(config.vocab_size, "vocab_size");
    require_positive(config.n_positions, "n_positions");
    require_positive(config.n_embd, "n_embd");
    require_positive(config.n_layer, "n_layer");
    require_positive(config.n_head, "n_head");

    if (config.n_embd % config.n_head != 0) {
        throw std::runtime_error(
            "config.json field 'n_embd' must be divisible by 'n_head'");
    }
    if (config.layer_norm_epsilon <= 0.0) {
        throw std::runtime_error(
            "config.json field 'layer_norm_epsilon' must be positive");
    }
    if (config.initializer_range <= 0.0) {
        throw std::runtime_error(
            "config.json field 'initializer_range' must be positive");
    }
}

}  // namespace

std::int64_t ModelConfig::head_size() const {
    return n_embd / n_head;
}

ModelConfig parse_model_config(std::string_view json_text) {
    Json json;
    try {
        json = Json::parse(json_text);
    } catch (const Json::parse_error& error) {
        throw std::runtime_error("could not parse config.json: " +
                                 std::string(error.what()));
    }

    const auto architectures = required<Json>(json, "architectures");
    if (!architectures.is_array() || architectures.empty()) {
        throw std::runtime_error(
            "config.json field 'architectures' must be a non-empty array");
    }

    std::string architecture;
    try {
        architecture = architectures.at(0).get<std::string>();
    } catch (const Json::exception& error) {
        throw std::runtime_error(
            "invalid config.json field 'architectures[0]': " +
            std::string(error.what()));
    }

    ModelConfig config{
        .architecture = std::move(architecture),
        .model_type = required<std::string>(json, "model_type"),
        .activation_function =
            required<std::string>(json, "activation_function"),
        .vocab_size = required<std::int64_t>(json, "vocab_size"),
        .n_positions = required<std::int64_t>(json, "n_positions"),
        .n_embd = required<std::int64_t>(json, "n_embd"),
        .n_layer = required<std::int64_t>(json, "n_layer"),
        .n_head = required<std::int64_t>(json, "n_head"),
        .bos_token_id = required<std::int64_t>(json, "bos_token_id"),
        .eos_token_id = required<std::int64_t>(json, "eos_token_id"),
        .layer_norm_epsilon =
            required<double>(json, "layer_norm_epsilon"),
        .initializer_range = required<double>(json, "initializer_range"),
    };

    validate(config);
    return config;
}

ModelConfig load_model_config(const std::filesystem::path& model_directory) {
    const auto config_path = model_directory / "config.json";
    std::ifstream input(config_path);
    if (!input) {
        throw std::runtime_error("could not open " + config_path.string());
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    if (input.bad()) {
        throw std::runtime_error("could not read " + config_path.string());
    }

    return parse_model_config(contents.str());
}

std::vector<TensorMetadata> load_safetensors_metadata(
    const std::filesystem::path& model_directory) {
    const auto path = model_directory / "model.safetensors";
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        throw std::runtime_error("could not open " + path.string());
    }

    std::array<unsigned char, 8> length_bytes{};
    input.read(reinterpret_cast<char*>(length_bytes.data()),
               static_cast<std::streamsize>(length_bytes.size()));

    if (!input) {
        throw std::runtime_error("could not read safetensors header length");
    }

    std::uint64_t header_length = 0;
    for (std::size_t i = 0; i < length_bytes.size(); ++i) {
        header_length |=
            std::uint64_t{length_bytes[i]} << (8 * i);
    }

    constexpr std::uint64_t kMaxHeaderSize = 100 * 1024 * 1024;
    const auto file_size = std::filesystem::file_size(path);

    if (header_length > kMaxHeaderSize ||
        header_length > file_size - length_bytes.size()) {
        throw std::runtime_error("invalid safetensors header length");
    }

    std::string header(static_cast<std::size_t>(header_length), '\0');
    input.read(header.data(),
               static_cast<std::streamsize>(header.size()));

    if (!input) {
        throw std::runtime_error("could not read safetensors metadata");
    }

    try {
        const auto json = Json::parse(header);
        std::vector<TensorMetadata> tensors;

        for (const auto& [name, value] : json.items()) {
            if (name == "__metadata__") {
                continue;
            }

            const auto offsets =
                value.at("data_offsets").get<std::array<std::uint64_t, 2>>();
            tensors.push_back({
                .name = name,
                .dtype = value.at("dtype").get<std::string>(),
                .shape = value.at("shape").get<std::vector<std::int64_t>>(),
                .data_begin = offsets[0],
                .data_end = offsets[1],
            });
        }

        return tensors;
    } catch (const Json::exception& error) {
        throw std::runtime_error(
            "could not parse safetensors metadata: " +
            std::string(error.what()));
    }
}

}  // namespace easyinfer
