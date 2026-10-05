#pragma once

#include <cstddef>
#include <string_view>

namespace easyinfer {

class Model {
public:
    virtual ~Model() = default;
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual std::size_t weight_count() const = 0;
    virtual void load_device_weights() = 0;
    virtual void forward() = 0;
};

}  // namespace easyinfer
