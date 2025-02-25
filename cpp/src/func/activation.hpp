#pragma once

#include "tensor.hpp"

namespace func {

float sigmoid(const float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(const float x) { return x * sigmoid(x); }

}  // namespace func