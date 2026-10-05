#pragma once

#include "easyinfer/kernels/context.hpp"

#include <cstddef>
#include <cstdint>

namespace easyinfer::kernels {

enum class WeightLayout {
    InputOutput, // [in_features, out_features], used by GPT-2 Conv1D weights
    OutputInput, // [out_features, in_features], used by ordinary linear layers
};

enum class RotaryLayout {
    HalfSplit,   // pairs the first half with the second half (Llama)
    Interleaved, // pairs adjacent values
};

// input: [tokens, in_features], output: [tokens, out_features]
void linear_f32(KernelContext& context,
                const float* input,
                const float* weight,
                const float* bias,
                float* output,
                std::int64_t tokens,
                std::int64_t in_features,
                std::int64_t out_features,
                WeightLayout weight_layout);

// token_ids: [batch, sequence], output: [batch, sequence, hidden]
// position_embedding may be null for models without learned position embeddings.
void embedding_f32(cudaStream_t stream,
                   const std::int32_t* token_ids,
                   const float* token_embedding,
                   const float* position_embedding,
                   float* output,
                   std::int64_t batch,
                   std::int64_t sequence,
                   std::int64_t hidden,
                   std::int64_t position_offset);

// All normalization tensors use [rows, hidden].
void layer_norm_f32(cudaStream_t stream,
                    const float* input,
                    const float* weight,
                    const float* bias,
                    float* output,
                    std::int64_t rows,
                    std::int64_t hidden,
                    float epsilon);

void rms_norm_f32(cudaStream_t stream,
                  const float* input,
                  const float* weight,
                  float* output,
                  std::int64_t rows,
                  std::int64_t hidden,
                  float epsilon);

void add_f32(cudaStream_t stream,
             const float* left,
             const float* right,
             float* output,
             std::size_t element_count);

void add_in_place_f32(cudaStream_t stream,
                      float* destination,
                      const float* source,
                      std::size_t element_count);

void add_bias_f32(cudaStream_t stream,
                  float* tensor,
                  const float* bias,
                  std::int64_t rows,
                  std::int64_t columns);

void gelu_f32(cudaStream_t stream, float* tensor, std::size_t element_count);
void silu_f32(cudaStream_t stream, float* tensor, std::size_t element_count);

// gate, value, and output are equally sized. output = silu(gate) * value.
void swiglu_f32(cudaStream_t stream,
                const float* gate,
                const float* value,
                float* output,
                std::size_t element_count);

// input: [batch, sequence, heads * head_dim]
// output: [batch, heads, sequence, head_dim]
void reshape_heads_f32(cudaStream_t stream,
                       const float* input,
                       float* output,
                       std::int64_t batch,
                       std::int64_t sequence,
                       std::int64_t heads,
                       std::int64_t head_dim);

// Reverses reshape_heads_f32.
void merge_heads_f32(cudaStream_t stream,
                     const float* input,
                     float* output,
                     std::int64_t batch,
                     std::int64_t sequence,
                     std::int64_t heads,
                     std::int64_t head_dim);

// packed_qkv: [batch, sequence, 3 * heads * head_dim]
// q/k/v: [batch, heads, sequence, head_dim]
void split_qkv_f32(cudaStream_t stream,
                   const float* packed_qkv,
                   float* query,
                   float* key,
                   float* value,
                   std::int64_t batch,
                   std::int64_t sequence,
                   std::int64_t heads,
                   std::int64_t head_dim);

// tensor: [batch, heads, sequence, head_dim], modified in place.
void apply_rope_f32(cudaStream_t stream,
                    float* tensor,
                    std::int64_t batch,
                    std::int64_t heads,
                    std::int64_t sequence,
                    std::int64_t head_dim,
                    std::int64_t position_offset,
                    float theta,
                    RotaryLayout layout = RotaryLayout::HalfSplit);

// scores: [batch, heads, query_length, key_length], modified in place.
void causal_softmax_f32(cudaStream_t stream,
                        float* scores,
                        std::int64_t batch,
                        std::int64_t heads,
                        std::int64_t query_length,
                        std::int64_t key_length,
                        std::int64_t past_tokens);

// q/k/v and output: [batch, heads, sequence, head_dim].
// scores is caller-owned scratch space with
// batch * heads * query_length * key_length floats.
// key_value_capacity is the physical cache stride; zero means key_length.
void causal_attention_f32(KernelContext& context,
                          const float* query,
                          const float* key,
                          const float* value,
                          float* scores,
                          float* output,
                          std::int64_t batch,
                          std::int64_t heads,
                          std::int64_t query_length,
                          std::int64_t key_length,
                          std::int64_t head_dim,
                          std::int64_t past_tokens,
                          std::int64_t key_value_capacity = 0);

// Copies [batch, heads, tokens, head_dim] into
// [batch, heads, cache_capacity, head_dim] at position_offset.
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
                        std::int64_t position_offset);

// logits: [rows, vocabulary], token_ids: [rows]
void argmax_f32(cudaStream_t stream,
                const float* logits,
                std::int32_t* token_ids,
                std::int64_t rows,
                std::int64_t vocabulary);

} // namespace easyinfer::kernels
