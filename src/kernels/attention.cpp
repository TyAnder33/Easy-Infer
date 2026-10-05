#include "easyinfer/kernels/ops.hpp"

#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace easyinfer::kernels {
namespace {

class MatmulDescriptor final {
public:
    MatmulDescriptor() {
        detail::check_cublas(
            cublasLtMatmulDescCreate(&value_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "create cuBLASLt matmul descriptor");
    }

    ~MatmulDescriptor() { cublasLtMatmulDescDestroy(value_); }
    MatmulDescriptor(const MatmulDescriptor&) = delete;
    MatmulDescriptor& operator=(const MatmulDescriptor&) = delete;
    [[nodiscard]] cublasLtMatmulDesc_t get() const noexcept { return value_; }

private:
    cublasLtMatmulDesc_t value_{};
};

class MatrixLayout final {
public:
    MatrixLayout(std::int64_t rows,
                 std::int64_t columns,
                 std::int64_t leading_dimension,
                 std::int32_t batch_count,
                 std::int64_t batch_stride) {
        detail::check_cublas(
            cublasLtMatrixLayoutCreate(
                &value_, CUDA_R_32F, rows, columns, leading_dimension),
            "create cuBLASLt matrix layout");

        try {
            const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
            detail::check_cublas(
                cublasLtMatrixLayoutSetAttribute(
                    value_,
                    CUBLASLT_MATRIX_LAYOUT_ORDER,
                    &order,
                    sizeof(order)),
                "set cuBLASLt row-major layout");

            if (batch_count > 1) {
                detail::check_cublas(
                    cublasLtMatrixLayoutSetAttribute(
                        value_,
                        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                        &batch_count,
                        sizeof(batch_count)),
                    "set cuBLASLt batch count");
                detail::check_cublas(
                    cublasLtMatrixLayoutSetAttribute(
                        value_,
                        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                        &batch_stride,
                        sizeof(batch_stride)),
                    "set cuBLASLt batch stride");
            }
        } catch (...) {
            cublasLtMatrixLayoutDestroy(value_);
            throw;
        }
    }

    ~MatrixLayout() { cublasLtMatrixLayoutDestroy(value_); }
    MatrixLayout(const MatrixLayout&) = delete;
    MatrixLayout& operator=(const MatrixLayout&) = delete;
    [[nodiscard]] cublasLtMatrixLayout_t get() const noexcept { return value_; }

private:
    cublasLtMatrixLayout_t value_{};
};

class MatmulPreference final {
public:
    explicit MatmulPreference(std::size_t workspace_bytes) {
        detail::check_cublas(cublasLtMatmulPreferenceCreate(&value_),
                             "create cuBLASLt preference");
        try {
            detail::check_cublas(
                cublasLtMatmulPreferenceSetAttribute(
                    value_,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspace_bytes,
                    sizeof(workspace_bytes)),
                "set cuBLASLt workspace size");
        } catch (...) {
            cublasLtMatmulPreferenceDestroy(value_);
            throw;
        }
    }

    ~MatmulPreference() { cublasLtMatmulPreferenceDestroy(value_); }
    MatmulPreference(const MatmulPreference&) = delete;
    MatmulPreference& operator=(const MatmulPreference&) = delete;
    [[nodiscard]] cublasLtMatmulPreference_t get() const noexcept {
        return value_;
    }

private:
    cublasLtMatmulPreference_t value_{};
};

struct MatrixShape {
    std::int64_t rows;
    std::int64_t columns;
    std::int64_t leading_dimension;
    std::int64_t batch_stride;
};

void matmul_f32(KernelContext& context,
                const float* left,
                const MatrixShape& left_shape,
                cublasOperation_t left_operation,
                const float* right,
                const MatrixShape& right_shape,
                cublasOperation_t right_operation,
                float* output,
                const MatrixShape& output_shape,
                std::int64_t batch_count,
                float alpha) {
    if (batch_count > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("cuBLASLt batch count is too large");
    }
    const auto layout_batch_count = static_cast<std::int32_t>(batch_count);
    MatmulDescriptor operation;

    detail::check_cublas(
        cublasLtMatmulDescSetAttribute(operation.get(),
                                      CUBLASLT_MATMUL_DESC_TRANSA,
                                      &left_operation,
                                      sizeof(left_operation)),
        "set cuBLASLt left transpose");
    detail::check_cublas(
        cublasLtMatmulDescSetAttribute(operation.get(),
                                      CUBLASLT_MATMUL_DESC_TRANSB,
                                      &right_operation,
                                      sizeof(right_operation)),
        "set cuBLASLt right transpose");

    MatrixLayout left_layout(left_shape.rows,
                             left_shape.columns,
                             left_shape.leading_dimension,
                             layout_batch_count,
                             left_shape.batch_stride);
    MatrixLayout right_layout(right_shape.rows,
                              right_shape.columns,
                              right_shape.leading_dimension,
                              layout_batch_count,
                              right_shape.batch_stride);
    MatrixLayout output_layout(output_shape.rows,
                               output_shape.columns,
                               output_shape.leading_dimension,
                               layout_batch_count,
                               output_shape.batch_stride);
    MatmulPreference preference(context.workspace_bytes());

    cublasLtMatmulHeuristicResult_t heuristic{};
    int result_count = 0;
    detail::check_cublas(
        cublasLtMatmulAlgoGetHeuristic(context.cublas_lt(),
                                      operation.get(),
                                      left_layout.get(),
                                      right_layout.get(),
                                      output_layout.get(),
                                      output_layout.get(),
                                      preference.get(),
                                      1,
                                      &heuristic,
                                      &result_count),
        "select cuBLASLt algorithm");
    if (result_count == 0) {
        throw std::runtime_error("cuBLASLt found no compatible matmul algorithm");
    }

    constexpr float beta = 0.0F;
    detail::check_cublas(
        cublasLtMatmul(context.cublas_lt(),
                      operation.get(),
                      &alpha,
                      left,
                      left_layout.get(),
                      right,
                      right_layout.get(),
                      &beta,
                      output,
                      output_layout.get(),
                      output,
                      output_layout.get(),
                      &heuristic.algo,
                      context.workspace(),
                      context.workspace_bytes(),
                      context.stream()),
        "run cuBLASLt matmul");
}

void require_positive(std::int64_t value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

} // namespace

void linear_f32(KernelContext& context,
                const float* input,
                const float* weight,
                const float* bias,
                float* output,
                std::int64_t tokens,
                std::int64_t in_features,
                std::int64_t out_features,
                WeightLayout weight_layout) {
    require_positive(tokens, "tokens");
    require_positive(in_features, "in_features");
    require_positive(out_features, "out_features");

    const MatrixShape input_shape{
        tokens, in_features, in_features, tokens * in_features};
    const MatrixShape output_shape{
        tokens, out_features, out_features, tokens * out_features};

    if (weight_layout == WeightLayout::InputOutput) {
        const MatrixShape weight_shape{in_features,
                                       out_features,
                                       out_features,
                                       in_features * out_features};
        matmul_f32(context,
                   input,
                   input_shape,
                   CUBLAS_OP_N,
                   weight,
                   weight_shape,
                   CUBLAS_OP_N,
                   output,
                   output_shape,
                   1,
                   1.0F);
        if (bias != nullptr) {
            add_bias_f32(
                context.stream(), output, bias, tokens, out_features);
        }
        return;
    }

    const MatrixShape weight_shape{out_features,
                                   in_features,
                                   in_features,
                                   in_features * out_features};
    matmul_f32(context,
               input,
               input_shape,
               CUBLAS_OP_N,
               weight,
               weight_shape,
               CUBLAS_OP_T,
               output,
               output_shape,
               1,
               1.0F);
    if (bias != nullptr) {
        add_bias_f32(context.stream(), output, bias, tokens, out_features);
    }
}

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
                          std::int64_t key_value_capacity) {
    require_positive(batch, "batch");
    require_positive(heads, "heads");
    require_positive(query_length, "query_length");
    require_positive(key_length, "key_length");
    require_positive(head_dim, "head_dim");
    if (past_tokens < 0) {
        throw std::invalid_argument("past_tokens cannot be negative");
    }
    if (key_value_capacity == 0) {
        key_value_capacity = key_length;
    }
    if (key_value_capacity < key_length) {
        throw std::invalid_argument(
            "key_value_capacity cannot be smaller than key_length");
    }

    const auto batch_heads = batch * heads;
    const MatrixShape query_shape{query_length,
                                  head_dim,
                                  head_dim,
                                  query_length * head_dim};
    const MatrixShape key_shape{
        key_length, head_dim, head_dim, key_value_capacity * head_dim};
    const MatrixShape score_shape{
        query_length, key_length, key_length, query_length * key_length};

    matmul_f32(context,
               query,
               query_shape,
               CUBLAS_OP_N,
               key,
               key_shape,
               CUBLAS_OP_T,
               scores,
               score_shape,
               batch_heads,
               1.0F / std::sqrt(static_cast<float>(head_dim)));

    causal_softmax_f32(context.stream(),
                       scores,
                       batch,
                       heads,
                       query_length,
                       key_length,
                       past_tokens);

    const MatrixShape value_shape{
        key_length, head_dim, head_dim, key_value_capacity * head_dim};
    const MatrixShape output_shape{query_length,
                                   head_dim,
                                   head_dim,
                                   query_length * head_dim};
    matmul_f32(context,
               scores,
               score_shape,
               CUBLAS_OP_N,
               value,
               value_shape,
               CUBLAS_OP_N,
               output,
               output_shape,
               batch_heads,
               1.0F);
}

} // namespace easyinfer::kernels
