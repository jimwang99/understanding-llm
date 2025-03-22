#pragma once

#include "logger.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void rms_norm_out(const Tensor<T> &ti, Tensor<T> &to, const float eps = 1e-5) {
  _ASSERT(ti.ndim() >= 1, ti.shape());
  _ASSERT(to.ndim() >= 1, to.shape());
  auto dim = ti.shape(0);
  auto batch = ti.size() / dim;

  _ASSERT(to.shape(0) == dim, to.shape());
  _ASSERT(to.size() == batch * dim, to.shape());

  for (size_t b = 0; b < batch; ++b) {
    // sum of x^2
    T sum = 0;
    for (size_t i = 0; i < dim; ++i) {
      sum += ti.at(i, b) * ti.at(i, b);
    }
    auto rms = std::sqrt(sum / dim + eps);
    // x / rms
    for (size_t i = 0; i < dim; ++i) {
      to.at(i, b) = ti.at(i, b) / rms;
    }
  }
};

}  // namespace func