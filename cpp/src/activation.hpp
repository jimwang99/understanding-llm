#pragma once

#include "logger.hpp"
#include "tensor.hpp"

namespace func {

float sigmoid(const float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(const float x) { return x * sigmoid(x); }

void sigmoid_inline(Tensor<float> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    x.at(i) = sigmoid(x.at(i));
  }
}

void sigmoid_out(const Tensor<float> &x, Tensor<float> &y) {
  ASSERT(y.shape() == x.shape(), "y.shape()={}, x.shape()={}", y.shape(),
         x.shape());
  for (size_t i = 0; i < x.size(); ++i) {
    y.at(i) = sigmoid(x.at(i));
  }
}

void silu_inline(Tensor<float> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    x.at(i) = silu(x.at(i));
  }
}

void silu_out(const Tensor<float> &x, Tensor<float> &y) {
  ASSERT(y.shape() == x.shape(), "y.shape()={}, x.shape()={}", y.shape(),
         x.shape());
  for (size_t i = 0; i < x.size(); ++i) {
    y.at(i) = silu(x.at(i));
  }
}

} // namespace func