#pragma once

#include "tensor.hpp"
#include <cassert>

namespace func {

template <typename T>
void rms_norm_inline(Tensor<T> &x, const float eps = 1e-5) {
  assert(x.ndim() >= 1);
  auto D = x.shape(0);
  auto B = x.size() / D;

  for (size_t b = 0; b < B; ++b) {
    // sum of x^2
    T sum = 0;
    for (size_t i = 0; i < D; ++i) {
      sum += x.at(i, b) * x.at(i, b);
    }
    auto rms = std::sqrt(sum / D + eps);
    // x / rms
    for (size_t i = 0; i < D; ++i) {
      x.at(i, b) = x.at(i, b) / rms;
    }
  }
};

} // namespace func