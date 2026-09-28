#include "easyinfer/weight_store.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace easyinfer {

WeightStore::~WeightStore() {
    if (mapped_file_ != nullptr) {
        ::munmap(mapped_file_, mapped_file_size_);
    }
}

void WeightStore::load_weights(
    const std::filesystem::path& model_directory,
    const std::vector<TensorMetadata>& metadata) {

    if (mapped_file_ != nullptr) {
        throw std::runtime_error("weights are already loaded");
    }

    const auto path = model_directory / "model.safetensors";
    const int file = ::open(path.c_str(), O_RDONLY);
    if (file == -1) {
        throw std::runtime_error("could not open " + path.string() + ": " +
                                 std::strerror(errno));
    }

    struct stat file_info {};
    if (::fstat(file, &file_info) == -1) {
        const std::string error = std::strerror(errno);
        ::close(file);
        throw std::runtime_error("could not inspect " + path.string() + ": " +
                                 error);
    }
    if (file_info.st_size < 8) {
        ::close(file);
        throw std::runtime_error("safetensors file is too small");
    }

    const auto file_size = static_cast<std::size_t>(file_info.st_size);
    void* mapping =
        ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file, 0);
    const int mmap_error = errno;
    ::close(file);

    if (mapping == MAP_FAILED) {
        throw std::runtime_error("could not map " + path.string() + ": " +
                                 std::strerror(mmap_error));
    }

    try {
        const auto* bytes = static_cast<const unsigned char*>(mapping);
        std::uint64_t header_length = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            header_length |= std::uint64_t{bytes[i]} << (8 * i);
        }

        if (header_length > file_size - 8) {
            throw std::runtime_error("invalid safetensors header length");
        }

        const auto data_start =
            8 + static_cast<std::size_t>(header_length);
        const auto data_size = file_size - data_start;
        const auto* tensor_data = bytes + data_start;

        std::unordered_map<std::string, TensorView> tensors;
        tensors.reserve(metadata.size());

        for (const auto& tensor : metadata) {
            if (tensor.dtype != "F32") {
                throw std::runtime_error("unsupported dtype for tensor '" +
                                         tensor.name + "': " + tensor.dtype);
            }
            if (tensor.data_begin > tensor.data_end ||
                tensor.data_end > data_size) {
                throw std::runtime_error("invalid data offsets for tensor '" +
                                         tensor.name + "'");
            }

            const auto byte_size = static_cast<std::size_t>(
                tensor.data_end - tensor.data_begin);
            const auto* host_data =
                tensor_data + static_cast<std::size_t>(tensor.data_begin);

            const auto [unused, inserted] = tensors.emplace(
                tensor.name,
                TensorView{
                    .host_data = host_data,
                    .dtype = DType::FP32,
                    .shape = tensor.shape,
                    .byte_size = byte_size,
                });
            static_cast<void>(unused);

            if (!inserted) {
                throw std::runtime_error("duplicate tensor name '" +
                                         tensor.name + "'");
            }
        }

        mapped_file_ = mapping;
        mapped_file_size_ = file_size;
        tensors_ = std::move(tensors);
    } catch (...) {
        ::munmap(mapping, file_size);
        throw;
    }
}


bool WeightStore::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const TensorView& WeightStore::at(const std::string& name) const {
    return tensors_.at(name);
}

std::size_t WeightStore::size() const {
    return tensors_.size();
}



}  // namespace easyinfer
