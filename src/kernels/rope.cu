#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace easyinfer::kernels {
namespace {

constexpr int threads_per_block = 256;

__global__ void rope_kernel(float* tensor,
                            std::int64_t sequence,
                            std::int64_t head_dim,
                            std::int64_t position_offset,
                            float theta,
                            RotaryLayout layout,
                            std::size_t pair_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= pair_count) {
        return;
    }

    const auto half_dim = head_dim / 2;
    const auto pair = static_cast<std::int64_t>(index % half_dim);
    const auto vector_index = static_cast<std::int64_t>(index / half_dim);
    const auto sequence_index = vector_index % sequence;
    const auto vector_start = vector_index * head_dim;

    std::int64_t first = 0;
    std::int64_t second = 0;
    if (layout == RotaryLayout::HalfSplit) {
        first = vector_start + pair;
        second = vector_start + pair + half_dim;
    } else {
        first = vector_start + 2 * pair;
        second = first + 1;
    }

    const auto position = position_offset + sequence_index;
    const float frequency =
        powf(theta, -2.0F * static_cast<float>(pair) /
                        static_cast<float>(head_dim));
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(static_cast<float>(position) * frequency, &sine, &cosine);

    const float first_value = tensor[first];
    const float second_value = tensor[second];
    tensor[first] = first_value * cosine - second_value * sine;
    tensor[second] = second_value * cosine + first_value * sine;
}

} // namespace

void apply_rope_f32(cudaStream_t stream,
                    float* tensor,
                    std::int64_t batch,
                    std::int64_t heads,
                    std::int64_t sequence,
                    std::int64_t head_dim,
                    std::int64_t position_offset,
                    float theta,
                    RotaryLayout layout) {
    if (batch <= 0 || heads <= 0 || sequence <= 0 || head_dim <= 0 ||
        head_dim % 2 != 0) {
        throw std::invalid_argument(
            "RoPE dimensions must be positive and head_dim must be even");
    }
    if (position_offset < 0 || theta <= 0.0F) {
        throw std::invalid_argument("invalid RoPE position offset or theta");
    }

    const auto pair_count = static_cast<std::size_t>(batch) *
                            static_cast<std::size_t>(heads) *
                            static_cast<std::size_t>(sequence) *
                            static_cast<std::size_t>(head_dim / 2);
    const auto blocks = static_cast<unsigned int>(
        (pair_count + threads_per_block - 1) / threads_per_block);
    rope_kernel<<<blocks, threads_per_block, 0, stream>>>(tensor,
                                                          sequence,
                                                          head_dim,
                                                          position_offset,
                                                          theta,
                                                          layout,
                                                          pair_count);
    detail::check_last_kernel("apply_rope_f32");
}

} // namespace easyinfer::kernels
