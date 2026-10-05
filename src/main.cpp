#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "easyinfer/model_config.hpp"
#include "easyinfer/model_registry.hpp"

namespace {

struct CliOptions {
    std::filesystem::path model_directory;
    std::filesystem::path output_path;
    std::vector<std::int32_t> input_ids;
    bool inspect{false};
    bool gpu{false};
};

[[noreturn]] void usage_error(std::string_view message) {
    throw std::runtime_error(std::string(message) +
                             "\nusage:\n"
                             "  engine --model <directory> --inspect [--gpu]\n"
                             "  engine --model <directory> --input-ids <id,id,...> "
                             "--output <file>");
}

std::vector<std::int32_t> parse_input_ids(std::string_view text) {
    std::vector<std::int32_t> result;
    std::size_t start = 0;

    while (start < text.size()) {
        const auto comma = text.find(',', start);
        const auto end = comma == std::string_view::npos ? text.size() : comma;
        std::int32_t token_id = 0;
        const auto [parsed_end, error] =
            std::from_chars(text.data() + start, text.data() + end, token_id);
        if (error != std::errc{} || parsed_end != text.data() + end ||
            token_id < 0) {
            usage_error("invalid --input-ids value");
        }
        result.push_back(token_id);
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
        if (start == text.size()) {
            usage_error("invalid --input-ids value");
        }
    }

    if (result.empty()) {
        usage_error("--input-ids cannot be empty");
    }
    return result;
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
        } else if (argument == "--gpu") {
            options.gpu = true;
        } else if (argument == "--input-ids") {
            if (++i >= argc) {
                usage_error("--input-ids requires comma-separated token IDs");
            }
            options.input_ids = parse_input_ids(argv[i]);
        } else if (argument == "--output") {
            if (++i >= argc) {
                usage_error("--output requires a file path");
            }
            options.output_path = argv[i];
        } else {
            usage_error("unknown argument: " + std::string(argument));
        }
    }

    if (options.model_directory.empty()) {
        usage_error("missing required --model argument");
    }
    const bool run_inference = !options.input_ids.empty();
    if (options.inspect == run_inference) {
        usage_error("choose either --inspect or --input-ids");
    }
    if (run_inference && options.output_path.empty()) {
        usage_error("inference requires --output");
    }
    if (options.inspect && !options.output_path.empty()) {
        usage_error("--output is only valid with --input-ids");
    }
    if (run_inference && options.gpu) {
        usage_error("inference uses the GPU automatically; omit --gpu");
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
        const auto safetensor_metadata =
            easyinfer::load_safetensors_metadata(options.model_directory);

        const auto model = easyinfer::create_model(
            config, options.model_directory, safetensor_metadata);

        if (!options.inspect) {
            model->load_device_weights();
            const auto logits = model->forward(options.input_ids);
            std::ofstream output(options.output_path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(logits.data()),
                         static_cast<std::streamsize>(logits.size() *
                                                      sizeof(float)));
            if (!output) {
                throw std::runtime_error("could not write " +
                                         options.output_path.string());
            }

            const auto next_token = static_cast<std::size_t>(
                std::max_element(logits.begin(), logits.end()) -
                logits.begin());
            std::cout << "Next token ID: " << next_token << '\n'
                      << "Wrote " << logits.size() << " logits to "
                      << options.output_path << '\n';
            return 0;
        }

        if (options.gpu) {
            model->load_device_weights();
        }

        print_config(config);
        print_safetensors_metadata(safetensor_metadata);
        std::cout << "\nSelected " << model->name() << '\n'
                  << "Validated " << model->weight_count() << " tensors\n";
        if (options.gpu) {
            std::cout << "Uploaded and bound GPU weights\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
