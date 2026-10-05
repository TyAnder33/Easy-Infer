#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cstdint>
#include <stdexcept>

namespace easyinfer::kernels {
namespace {

constexpr int threads_per_block = 256;

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

__global__ void layer_norm_kernel(const float* input,
                                  const float* weight,
                                  const float* bias,
                                  float* output,
                                  std::int64_t hidden,
                                  float epsilon) {
    extern __shared__ float scratch[];
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto offset = row * hidden;

    float sum = 0.0F;
    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < hidden;
         column += blockDim.x) {
        sum += input[offset + column];
    }
    const float mean = block_sum(sum, scratch) / static_cast<float>(hidden);

    float squared_difference_sum = 0.0F;
    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < hidden;
         column += blockDim.x) {
        const float difference = input[offset + column] - mean;
        squared_difference_sum += difference * difference;
    }
    const float variance = block_sum(squared_difference_sum, scratch) /
                           static_cast<float>(hidden);
    const float inverse_standard_deviation = rsqrtf(variance + epsilon);

    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < hidden;
         column += blockDim.x) {
        const float normalized =
            (input[offset + column] - mean) * inverse_standard_deviation;
        output[offset + column] = normalized * weight[column] + bias[column];
    }
}

__global__ void rms_norm_kernel(const float* input,
                                const float* weight,
                                float* output,
                                std::int64_t hidden,
                                float epsilon) {
    extern __shared__ float scratch[];
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto offset = row * hidden;

    float squared_sum = 0.0F;
    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < hidden;
         column += blockDim.x) {
        const float value = input[offset + column];
        squared_sum += value * value;
    }
    const float mean_square =
        block_sum(squared_sum, scratch) / static_cast<float>(hidden);
    const float inverse_rms = rsqrtf(mean_square + epsilon);

    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < hidden;
         column += blockDim.x) {
        output[offset + column] =
            input[offset + column] * inverse_rms * weight[column];
    }
}

void validate_dimensions(std::int64_t rows,
                         std::int64_t hidden,
                         float epsilon) {
    if (rows <= 0 || hidden <= 0) {
        throw std::invalid_argument("normalization dimensions must be positive");
    }
    if (epsilon <= 0.0F) {
        throw std::invalid_argument("normalization epsilon must be positive");
    }
}

} // namespace

void layer_norm_f32(cudaStream_t stream,
                    const float* input,
                    const float* weight,
                    const float* bias,
                    float* output,
                    std::int64_t rows,
                    std::int64_t hidden,
                    float epsilon) {
    validate_dimensions(rows, hidden, epsilon);
    layer_norm_kernel<<<static_cast<unsigned int>(rows),
                        threads_per_block,
                        threads_per_block * sizeof(float),
                        stream>>>(input, weight, bias, output, hidden, epsilon);
    detail::check_last_kernel("layer_norm_f32");
}

void rms_norm_f32(cudaStream_t stream,
                  const float* input,
                  const float* weight,
                  float* output,
                  std::int64_t rows,
                  std::int64_t hidden,
                  float epsilon) {
    validate_dimensions(rows, hidden, epsilon);
    rms_norm_kernel<<<static_cast<unsigned int>(rows),
                      threads_per_block,
                      threads_per_block * sizeof(float),
                      stream>>>(input, weight, output, hidden, epsilon);
    detail::check_last_kernel("rms_norm_f32");
}

} // namespace easyinfer::kernels
