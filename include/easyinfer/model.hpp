#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace easyinfer {

class Model {
public:
    virtual ~Model() = default;
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual std::size_t weight_count() const = 0;
    virtual void load_device_weights() = 0;
    [[nodiscard]] virtual std::vector<float> forward(
        const std::vector<std::int32_t>& input_ids) = 0;
};

}  // namespace easyinfer
