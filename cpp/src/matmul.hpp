#pragma once

#include <cassert>

#include "tensor.hpp"

namespace func {

template <typename T, bool IMPLICIT_TRANSPOSE = false>
void matmul_2d_out(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &z) {
  auto logger = setup_logger();
  ASSERT(x.ndim() == 3, x.shape());
  ASSERT(y.ndim() == 3, y.shape());
  ASSERT(z.ndim() == 3, z.shape());
  ASSERT(x.shape(2) == y.shape(2), "x.shape()={}, y.shape()={}", x.shape(),
         y.shape());
  ASSERT(x.shape(2) == z.shape(2), "x.shape()={}, z.shape()={}", x.shape(),
         z.shape());

  auto K = x.shape(0);
  auto N = x.shape(1);
  auto B = x.shape(2);
  size_t M;
  if constexpr (!IMPLICIT_TRANSPOSE) {
    M = y.shape(0);
    ASSERT(K == y.shape(1), "K={}, y.shape()={}", K, y.shape());
  } else {
    M = y.shape(1);
    ASSERT(K == y.shape(0), "K={}, y.shape()={}", K, y.shape());
  }
  ASSERT(z.shape(0) == M, "M={}, z.shape()={}", M, z.shape());
  ASSERT(z.shape(1) == N, "N={}, z.shape()={}", N, z.shape());

  TRACE("B={} N={} M={} K={}", B, N, M, K);

  z.zeros();
  for (size_t b = 0; b < B; ++b) {
    for (size_t n = 0; n < N; ++n) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
          if constexpr (!IMPLICIT_TRANSPOSE) {
            z.at(m, n, b) += x.at(k, n, b) * y.at(m, k, b);
          } else {
            z.at(m, n, b) += x.at(k, n, b) * y.at(k, m, b);
          }
        }
      }
    }
  }
}

} // namespace func