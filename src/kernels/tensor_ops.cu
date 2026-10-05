#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace easyinfer::kernels {
namespace {

constexpr int threads_per_block = 256;

__global__ void add_kernel(const float* left,
                           const float* right,
                           float* output,
                           std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < element_count) {
        output[index] = left[index] + right[index];
    }
}

__global__ void add_in_place_kernel(float* destination,
                                    const float* source,
                                    std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < element_count) {
        destination[index] += source[index];
    }
}

__global__ void add_bias_kernel(float* tensor,
                                const float* bias,
                                std::int64_t columns,
                                std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < element_count) {
        tensor[index] += bias[index % static_cast<std::size_t>(columns)];
    }
}

__global__ void gelu_kernel(float* tensor, std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < element_count) {
        constexpr float coefficient = 0.7978845608028654F;
        const float value = tensor[index];
        tensor[index] =
            0.5F * value *
            (1.0F + tanhf(coefficient *
                          (value + 0.044715F * value * value * value)));
    }
}

__device__ float silu(float value) {
    return value / (1.0F + expf(-value));
}

__global__ void silu_kernel(float* tensor, std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < element_count) {
        tensor[index] = silu(tensor[index]);
    }
}

__global__ void swiglu_kernel(const float* gate,
                              const float* value,
                              float* output,
                              std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < element_count) {
        output[index] = silu(gate[index]) * value[index];
    }
}

__global__ void argmax_kernel(const float* logits,
                              std::int32_t* token_ids,
                              std::int64_t vocabulary) {
    extern __shared__ unsigned char shared_memory[];
    auto* values = reinterpret_cast<float*>(shared_memory);
    auto* indices = reinterpret_cast<std::int32_t*>(
        values + static_cast<std::size_t>(blockDim.x));

    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto row_offset = row * vocabulary;
    float best_value = -FLT_MAX;
    std::int32_t best_index = 0;

    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < vocabulary;
         column += blockDim.x) {
        const float candidate = logits[row_offset + column];
        if (candidate > best_value) {
            best_value = candidate;
            best_index = static_cast<std::int32_t>(column);
        }
    }

    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const auto other_index = indices[threadIdx.x + stride];
            if (other_value > values[threadIdx.x] ||
                (other_value == values[threadIdx.x] &&
                 other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        token_ids[row] = indices[0];
    }
}

unsigned int block_count(std::size_t element_count) {
    if (element_count == 0) {
        throw std::invalid_argument("element_count must be positive");
    }
    return static_cast<unsigned int>(
        (element_count + threads_per_block - 1) / threads_per_block);
}

} // namespace

void add_f32(cudaStream_t stream,
             const float* left,
             const float* right,
             float* output,
             std::size_t element_count) {
    add_kernel<<<block_count(element_count), threads_per_block, 0, stream>>>(
        left, right, output, element_count);
    detail::check_last_kernel("add_f32");
}

void add_in_place_f32(cudaStream_t stream,
                      float* destination,
                      const float* source,
                      std::size_t element_count) {
    add_in_place_kernel<<<block_count(element_count),
                          threads_per_block,
                          0,
                          stream>>>(destination, source, element_count);
    detail::check_last_kernel("add_in_place_f32");
}

void add_bias_f32(cudaStream_t stream,
                  float* tensor,
                  const float* bias,
                  std::int64_t rows,
                  std::int64_t columns) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("bias dimensions must be positive");
    }
    const auto element_count = static_cast<std::size_t>(rows) *
                               static_cast<std::size_t>(columns);
    add_bias_kernel<<<block_count(element_count),
                      threads_per_block,
                      0,
                      stream>>>(tensor, bias, columns, element_count);
    detail::check_last_kernel("add_bias_f32");
}

void gelu_f32(cudaStream_t stream, float* tensor, std::size_t element_count) {
    gelu_kernel<<<block_count(element_count), threads_per_block, 0, stream>>>(
        tensor, element_count);
    detail::check_last_kernel("gelu_f32");
}

void silu_f32(cudaStream_t stream, float* tensor, std::size_t element_count) {
    silu_kernel<<<block_count(element_count), threads_per_block, 0, stream>>>(
        tensor, element_count);
    detail::check_last_kernel("silu_f32");
}

void swiglu_f32(cudaStream_t stream,
                const float* gate,
                const float* value,
                float* output,
                std::size_t element_count) {
    swiglu_kernel<<<block_count(element_count), threads_per_block, 0, stream>>>(
        gate, value, output, element_count);
    detail::check_last_kernel("swiglu_f32");
}

void argmax_f32(cudaStream_t stream,
                const float* logits,
                std::int32_t* token_ids,
                std::int64_t rows,
                std::int64_t vocabulary) {
    if (rows <= 0 || vocabulary <= 0) {
        throw std::invalid_argument("argmax dimensions must be positive");
    }
    const auto shared_bytes = threads_per_block *
                              (sizeof(float) + sizeof(std::int32_t));
    argmax_kernel<<<static_cast<unsigned int>(rows),
                    threads_per_block,
                    shared_bytes,
                    stream>>>(logits, token_ids, vocabulary);
    detail::check_last_kernel("argmax_f32");
}

} // namespace easyinfer::kernels
