#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace easyinfer::kernels {
namespace {

constexpr int threads_per_block = 256;

__device__ float block_max(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] =
                fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float block_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__global__ void causal_softmax_kernel(float* scores,
                                      std::int64_t query_length,
                                      std::int64_t key_length,
                                      std::int64_t past_tokens) {
    extern __shared__ float scratch[];
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto query = row % query_length;
    const auto last_visible_key = past_tokens + query;
    const auto row_offset = row * key_length;

    float local_maximum = -FLT_MAX;
    for (auto key = static_cast<std::int64_t>(threadIdx.x);
         key < key_length;
         key += blockDim.x) {
        if (key <= last_visible_key) {
            local_maximum = fmaxf(local_maximum, scores[row_offset + key]);
        }
    }
    const float maximum = block_max(local_maximum, scratch);

    float local_sum = 0.0F;
    for (auto key = static_cast<std::int64_t>(threadIdx.x);
         key < key_length;
         key += blockDim.x) {
        float probability = 0.0F;
        if (key <= last_visible_key) {
            probability = expf(scores[row_offset + key] - maximum);
        }
        scores[row_offset + key] = probability;
        local_sum += probability;
    }
    const float sum = block_sum(local_sum, scratch);

    for (auto key = static_cast<std::int64_t>(threadIdx.x);
         key < key_length;
         key += blockDim.x) {
        scores[row_offset + key] /= sum;
    }
}

} // namespace

void causal_softmax_f32(cudaStream_t stream,
                        float* scores,
                        std::int64_t batch,
                        std::int64_t heads,
                        std::int64_t query_length,
                        std::int64_t key_length,
                        std::int64_t past_tokens) {
    if (batch <= 0 || heads <= 0 || query_length <= 0 || key_length <= 0) {
        throw std::invalid_argument("attention dimensions must be positive");
    }
    if (past_tokens < 0 || past_tokens >= key_length) {
        throw std::invalid_argument("past_tokens is outside the key sequence");
    }

    const auto rows = batch * heads * query_length;
    causal_softmax_kernel<<<static_cast<unsigned int>(rows),
                            threads_per_block,
                            threads_per_block * sizeof(float),
                            stream>>>(scores,
                                     query_length,
                                     key_length,
                                     past_tokens);
    detail::check_last_kernel("causal_softmax_f32");
}

} // namespace easyinfer::kernels
