#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace easyinfer::kernels {
namespace {

constexpr int threads_per_block = 256;

__global__ void reshape_heads_kernel(const float* input,
                                     float* output,
                                     std::int64_t sequence,
                                     std::int64_t heads,
                                     std::int64_t head_dim,
                                     std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= element_count) {
        return;
    }

    auto remaining = static_cast<std::int64_t>(index);
    const auto dimension = remaining % head_dim;
    remaining /= head_dim;
    const auto sequence_index = remaining % sequence;
    remaining /= sequence;
    const auto head = remaining % heads;
    const auto batch = remaining / heads;

    const auto input_index =
        ((batch * sequence + sequence_index) * heads + head) * head_dim +
        dimension;
    output[index] = input[input_index];
}

__global__ void merge_heads_kernel(const float* input,
                                   float* output,
                                   std::int64_t sequence,
                                   std::int64_t heads,
                                   std::int64_t head_dim,
                                   std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= element_count) {
        return;
    }

    auto remaining = static_cast<std::int64_t>(index);
    const auto dimension = remaining % head_dim;
    remaining /= head_dim;
    const auto head = remaining % heads;
    remaining /= heads;
    const auto sequence_index = remaining % sequence;
    const auto batch = remaining / sequence;

    const auto input_index =
        ((batch * heads + head) * sequence + sequence_index) * head_dim +
        dimension;
    output[index] = input[input_index];
}

__global__ void split_qkv_kernel(const float* packed_qkv,
                                 float* query,
                                 float* key,
                                 float* value,
                                 std::int64_t sequence,
                                 std::int64_t heads,
                                 std::int64_t head_dim,
                                 std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= element_count) {
        return;
    }

    auto remaining = static_cast<std::int64_t>(index);
    const auto dimension = remaining % head_dim;
    remaining /= head_dim;
    const auto sequence_index = remaining % sequence;
    remaining /= sequence;
    const auto head = remaining % heads;
    const auto batch = remaining / heads;

    const auto hidden = heads * head_dim;
    const auto packed_base = (batch * sequence + sequence_index) * 3 * hidden;
    const auto packed_column = head * head_dim + dimension;
    query[index] = packed_qkv[packed_base + packed_column];
    key[index] = packed_qkv[packed_base + hidden + packed_column];
    value[index] = packed_qkv[packed_base + 2 * hidden + packed_column];
}

__global__ void write_kv_cache_kernel(const float* key,
                                      const float* value,
                                      float* key_cache,
                                      float* value_cache,
                                      std::int64_t heads,
                                      std::int64_t tokens,
                                      std::int64_t head_dim,
                                      std::int64_t cache_capacity,
                                      std::int64_t position_offset,
                                      std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= element_count) {
        return;
    }

    auto remaining = static_cast<std::int64_t>(index);
    const auto dimension = remaining % head_dim;
    remaining /= head_dim;
    const auto token = remaining % tokens;
    remaining /= tokens;
    const auto head = remaining % heads;
    const auto batch = remaining / heads;

    const auto cache_index =
        ((batch * heads + head) * cache_capacity + position_offset + token) *
            head_dim +
        dimension;
    key_cache[cache_index] = key[index];
    value_cache[cache_index] = value[index];
}

std::size_t element_count(std::int64_t batch,
                          std::int64_t sequence,
                          std::int64_t heads,
                          std::int64_t head_dim) {
    if (batch <= 0 || sequence <= 0 || heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("tensor dimensions must be positive");
    }
    return static_cast<std::size_t>(batch) *
           static_cast<std::size_t>(sequence) *
           static_cast<std::size_t>(heads) *
           static_cast<std::size_t>(head_dim);
}

unsigned int block_count(std::size_t count) {
    return static_cast<unsigned int>(
        (count + threads_per_block - 1) / threads_per_block);
}

} // namespace

void reshape_heads_f32(cudaStream_t stream,
                       const float* input,
                       float* output,
                       std::int64_t batch,
                       std::int64_t sequence,
                       std::int64_t heads,
                       std::int64_t head_dim) {
    const auto count = element_count(batch, sequence, heads, head_dim);
    reshape_heads_kernel<<<block_count(count), threads_per_block, 0, stream>>>(
        input, output, sequence, heads, head_dim, count);
    detail::check_last_kernel("reshape_heads_f32");
}

void merge_heads_f32(cudaStream_t stream,
                     const float* input,
                     float* output,
                     std::int64_t batch,
                     std::int64_t sequence,
                     std::int64_t heads,
                     std::int64_t head_dim) {
    const auto count = element_count(batch, sequence, heads, head_dim);
    merge_heads_kernel<<<block_count(count), threads_per_block, 0, stream>>>(
        input, output, sequence, heads, head_dim, count);
    detail::check_last_kernel("merge_heads_f32");
}

void split_qkv_f32(cudaStream_t stream,
                   const float* packed_qkv,
                   float* query,
                   float* key,
                   float* value,
                   std::int64_t batch,
                   std::int64_t sequence,
                   std::int64_t heads,
                   std::int64_t head_dim) {
    const auto count = element_count(batch, sequence, heads, head_dim);
    split_qkv_kernel<<<block_count(count), threads_per_block, 0, stream>>>(
        packed_qkv,
        query,
        key,
        value,
        sequence,
        heads,
        head_dim,
        count);
    detail::check_last_kernel("split_qkv_f32");
}

void write_kv_cache_f32(cudaStream_t stream,
                        const float* key,
                        const float* value,
                        float* key_cache,
                        float* value_cache,
                        std::int64_t batch,
                        std::int64_t heads,
                        std::int64_t tokens,
                        std::int64_t head_dim,
                        std::int64_t cache_capacity,
                        std::int64_t position_offset) {
    const auto count = element_count(batch, tokens, heads, head_dim);
    if (cache_capacity <= 0 || position_offset < 0 ||
        position_offset + tokens > cache_capacity) {
        throw std::invalid_argument("KV cache range is out of bounds");
    }
    write_kv_cache_kernel<<<block_count(count), threads_per_block, 0, stream>>>(
        key,
        value,
        key_cache,
        value_cache,
        heads,
        tokens,
        head_dim,
        cache_capacity,
        position_offset,
        count);
    detail::check_last_kernel("write_kv_cache_f32");
}

} // namespace easyinfer::kernels
