#pragma once

namespace easyinfer {

class Model {
public:
    virtual ~Model() = default;
    virtual void forward() = 0;
};

}  // namespace easyinfer
