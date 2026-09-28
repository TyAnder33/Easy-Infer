#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "easyinfer/model_config.hpp"
#include "easyinfer/weight_store.hpp"

namespace {

struct CliOptions {
    std::filesystem::path model_directory;
    bool inspect{false};
};

[[noreturn]] void usage_error(std::string_view message) {
    throw std::runtime_error(std::string(message) +
                             "\nusage: engine --model <directory> --inspect");
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string_view argument{argv[i]};
        if (argument == "--model") {
            if (++i >= argc) {
                usage_error("--model requires a directory");
            }
            options.model_directory = argv[i];
        } else if (argument == "--inspect") {
            options.inspect = true;
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }

    if (options.model_directory.empty()) {
        usage_error("missing required --model argument");
    }
    if (!options.inspect) {
        usage_error("this milestone only supports --inspect");
    }
    return options;
}

void print_config(const easyinfer::ModelConfig& config) {
    std::cout << "Loaded GPT-2 configuration\n"
              << "  architecture:       " << config.architecture << '\n'
              << "  model_type:         " << config.model_type << '\n'
              << "  vocab_size:         " << config.vocab_size << '\n'
              << "  n_positions:        " << config.n_positions << '\n'
              << "  n_layer:            " << config.n_layer << '\n'
              << "  n_head:             " << config.n_head << '\n'
              << "  n_embd:             " << config.n_embd << '\n'
              << "  head_size:          " << config.head_size() << '\n'
              << "  activation_function:" << ' '
              << config.activation_function << '\n'
              << "  layer_norm_epsilon: " << config.layer_norm_epsilon << '\n'
              << "  initializer_range:  " << config.initializer_range << '\n'
              << "  bos_token_id:       " << config.bos_token_id << '\n'
              << "  eos_token_id:       " << config.eos_token_id << '\n';
}

void print_safetensors_metadata(
    const std::vector<easyinfer::TensorMetadata>& tensors) {
    std::cout << "\nSafetensors (" << tensors.size() << " tensors)\n";

    for (const auto& tensor : tensors) {
        std::cout << "  " << tensor.name << "  " << tensor.dtype << "  [";

        for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << tensor.shape[i];
        }

        std::cout << "]\n";
    }
}

void validate_tensors(const easyinfer::ModelConfig& config,
                      const easyinfer::WeightStore& weights) {
    std::size_t expected_count = 0;

    const auto expect = [&](const std::string& name,
                            std::vector<std::int64_t> shape) {
        if (!weights.contains(name)) {
            throw std::runtime_error("missing tensor '" + name + "'");
        }

        const auto& tensor = weights.at(name);
        if (tensor.dtype != easyinfer::DType::FP32) {
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

    const auto hidden = config.n_embd;
    const auto mlp_hidden = 4 * hidden;

    expect("wte.weight", {config.vocab_size, hidden});
    expect("wpe.weight", {config.n_positions, hidden});
    expect("ln_f.weight", {hidden});
    expect("ln_f.bias", {hidden});

    for (std::int64_t layer = 0; layer < config.n_layer; ++layer) {
        const std::string prefix = "h." + std::to_string(layer) + ".";

        expect(prefix + "attn.bias",
               {1, 1, config.n_positions, config.n_positions});
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

    if (weights.size() != expected_count) {
        throw std::runtime_error("checkpoint contains unexpected tensors");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {

        const auto options = parse_cli(argc, argv);
        const auto config =
            easyinfer::load_model_config(options.model_directory);
        
        const auto safetensor_metadata = easyinfer::load_safetensors_metadata(options.model_directory);

        easyinfer::WeightStore weights;
        weights.load_weights(options.model_directory, safetensor_metadata);
        validate_tensors(config, weights);

    
        print_config(config);
        print_safetensors_metadata(safetensor_metadata);
        std::cout << "\nValidated " << weights.size() << " tensors\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
