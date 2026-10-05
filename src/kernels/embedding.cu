#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace easyinfer::kernels {
namespace {

constexpr int threads_per_block = 256;

__global__ void embedding_kernel(const std::int32_t* token_ids,
                                 const float* token_embedding,
                                 const float* position_embedding,
                                 float* output,
                                 std::int64_t sequence,
                                 std::int64_t hidden,
                                 std::int64_t position_offset,
                                 std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= element_count) {
        return;
    }

    const auto hidden_index = static_cast<std::int64_t>(index % hidden);
    const auto token_index = static_cast<std::int64_t>(index / hidden);
    const auto sequence_index = token_index % sequence;
    const auto token_id = static_cast<std::int64_t>(token_ids[token_index]);

    float value = token_embedding[token_id * hidden + hidden_index];
    if (position_embedding != nullptr) {
        const auto position = position_offset + sequence_index;
        value += position_embedding[position * hidden + hidden_index];
    }
    output[index] = value;
}

} // namespace

void embedding_f32(cudaStream_t stream,
                   const std::int32_t* token_ids,
                   const float* token_embedding,
                   const float* position_embedding,
                   float* output,
                   std::int64_t batch,
                   std::int64_t sequence,
                   std::int64_t hidden,
                   std::int64_t position_offset) {
    if (batch <= 0 || sequence <= 0 || hidden <= 0) {
        throw std::invalid_argument("embedding dimensions must be positive");
    }
    if (position_offset < 0) {
        throw std::invalid_argument("position_offset cannot be negative");
    }

    const auto element_count = static_cast<std::size_t>(batch) *
                               static_cast<std::size_t>(sequence) *
                               static_cast<std::size_t>(hidden);
    const auto blocks = static_cast<unsigned int>(
        (element_count + threads_per_block - 1) / threads_per_block);
    embedding_kernel<<<blocks, threads_per_block, 0, stream>>>(token_ids,
                                                               token_embedding,
                                                               position_embedding,
                                                               output,
                                                               sequence,
                                                               hidden,
                                                               position_offset,
                                                               element_count);
    detail::check_last_kernel("embedding_f32");
}

} // namespace easyinfer::kernels
