#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "easyinfer/model_config.hpp"
#include "easyinfer/model_registry.hpp"

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

}  // namespace

int main(int argc, char** argv) {
    try {

        const auto options = parse_cli(argc, argv);
        const auto config =
            easyinfer::load_model_config(options.model_directory);
        
        const auto safetensor_metadata = easyinfer::load_safetensors_metadata(options.model_directory);

        const auto model = easyinfer::create_model(
            config, options.model_directory, safetensor_metadata);

    
        print_config(config);
        print_safetensors_metadata(safetensor_metadata);
        std::cout << "\nSelected " << model->name() << '\n'
                  << "Validated " << model->weight_count() << " tensors\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
