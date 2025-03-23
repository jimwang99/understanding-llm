#pragma once

#include "logger.hpp"
#include "tensor.hpp"

namespace func {

template <typename T> T max_out(const Tensor<T> &in, Tensor<T> &out) {
  ASSERT(in.ndim() == 2, in.shape());
  ASSERT(out.ndim() == 1, out.shape());
  ASSERT(in.shape(1) == out.shape(0), "in.shape()={} out.shape()={}",
         in.shape(), out.shape());

  auto D = in.shape(0);
  auto B = in.shape(1);

  for (size_t b = 0; b < B; ++b) {
    out.at(b) = in.at(0, b);
    for (size_t d = 1; d < D; ++d) {
      if (in.at(d, b) > out.at(b)) {
        out.at(b) = in.at(d, b);
      }
    }
  }
}

template <typename T> void sum_out(const Tensor<T> &in, Tensor<T> &out) {
  ASSERT(in.ndim() == 2, in.shape());
  ASSERT(out.ndim() == 1, out.shape());
  ASSERT(in.shape(1) == out.shape(0), "in.shape()={} out.shape()={}",
         in.shape(), out.shape());

  auto D = in.shape(0);
  auto B = in.shape(1);

  for (size_t b = 0; b < B; ++b) {
    out.at(b) = 0;
    for (size_t d = 0; d < D; ++d) {
      out.at(b) += in.at(d, b);
    }
  }
}

template <typename T>
void softmax_inline(Tensor<T> &x, const bool safe_softmax = true) {
  auto shape = x.shape();
  x.view({shape[0], x.size() / shape[0]});
  auto D = x.shape(0);
  auto B = x.shape(1);

  // do it batch by batch to better utilize cache
  for (size_t b = 0; b < B; ++b) {
    // max(x)
    T max = 0;
    if (safe_softmax) {
      max = x.at(0, b);
      for (size_t d = 1; d < D; ++d) {
        if (x.at(d, b) > max) {
          max = x.at(d, b);
        }
      }
    }

    // x = exp(x - max)
    T sum = 0;
    for (size_t d = 0; d < D; ++d) {
      x.at(d, b) = std::exp(x.at(d, b) - max);
      sum += x.at(d, b);
    }

    // x = x / sum
    for (size_t d = 0; d < D; ++d) {
      x.at(d, b) /= sum;
    }
  }
  x.view(shape);
}

} // namespace func