#pragma once

#include "tensor.hpp"
#include <cassert>

namespace func {

template <typename T>
void softmax_inline(Tensor<T> &x, const bool safe_softmax = true) {
  assert(x.shape().size() >= 1);
  auto D = x.shape(0);
  auto B = x.size() / D;

  for (size_t b = 0; b < B; ++b) {
    T max = 0;
    if (safe_softmax) {
      // max(x)
      max = x.at(b, 0);
      for (size_t d = 1; d < D; ++d) {
        if (x.at(b, d) > max) {
          max = x.at(b, d);
        }
      }
    }
    // exp(x - max)
    T sum = 0;
    for (size_t d = 0; d < D; ++d) {
      x.at(b, d) = std::exp(x.at(b, d) - max);
      sum += x.at(b, d);
    }
    // exp(x - max) / sum
    for (size_t d = 0; d < D; ++d) {
      x.at(b, d) /= sum;
    }
  }
}

} // namespace func