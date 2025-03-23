#pragma once

#include "tensor.hpp"

namespace func {

template <typename T>
void argmax_out(const Tensor<T> &input, Tensor<size_t> &output) {
  ASSERT(input.ndim() == 3, input.shape());
  ASSERT(output.ndim() == 2, output.shape());
  ASSERT(input.shape(0) == output.shape(0),
         "input.shape()={} output.shape()={}", input.shape(), output.shape());
  ASSERT(input.shape(2) == output.shape(1),
         "input.shape()={} output.shape()={}", input.shape(), output.shape());
  auto D0 = input.shape(0);
  auto D1 = input.shape(1);
  auto D2 = input.shape(2);
  for (auto i2 = 0; i2 < D2; ++i2) {
    for (auto i0 = 0; i0 < D0; ++i0) {
      T max_val = input.at(i0, 0, i2);
      auto max_idx = 0;
      for (auto i1 = 1; i1 < D1; ++i1) {
        if (input.at(i0, i1, i2) > max_val) {
          max_val = input.at(i0, i1, i2);
          max_idx = i1;
        }
      }
      output.at(i0, i2) = max_idx;
    }
  }
}

template <typename T>
void sampling_greedy(Tensor<T> &logit, Tensor<size_t> &token) {
  ASSERT(logit.ndim() == 3, logit.shape());
  ASSERT(token.ndim() == 2, token.shape());

  auto B = logit.shape(2);
  auto L = logit.shape(1);
  auto V = logit.shape(0);

  ASSERT(token.shape(1) == B, "token.shape()={} logit.shape()={}",
         token.shape(), logit.shape());
  ASSERT(token.shape(0) == L, "token.shape()={} logit.shape()={}",
         token.shape(), logit.shape());

  logit.view({1, V, B * L});
  token.view({1, B * L});
  argmax_out<T>(logit, token);
  logit.view({V, L, B});
  token.view({L, B});
}

// template <typename T, int K>
// void sampling_top_k(Tensor<T> &logit, Tensor<T> &token, float temperature) {}

// template <typename T, int K>
// void sampling_top_p(const Tensor<T> &logit, Tensor<T> &token) {}

} // namespace func