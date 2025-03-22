#pragma once

#include <cassert>

#include "tensor.hpp"

namespace func {

template <typename T, bool IMPLICIT_TRANSPOSE = false>
void matmul_2d_out(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &z) {
  ASSERT(x.ndim() == 3, x.shape());
  ASSERT(y.ndim() == 3, y.shape());
  ASSERT(z.ndim() == 3, z.shape());
  ASSERT(x.shape(2) == y.shape(2),
         fmt::format("x.shape()={}, y.shape()={}", x.shape(), y.shape()));
  ASSERT(x.shape(2) == z.shape(2),
         fmt::format("x.shape()={}, z.shape()={}", x.shape(), z.shape()));

  auto K = x.shape(0);
  auto N = x.shape(1);
  auto B = x.shape(2);
  size_t M;
  if (!IMPLICIT_TRANSPOSE) {
    M = y.shape(0);
    assert(K == y.shape(1));
  } else {
    M = y.shape(1);
    assert(K == y.shape(0));
  }
  assert(z.shape(0) == M);
  assert(z.shape(1) == N);

  z.zeros();
  for (size_t b = 0; b < B, ++b) {
    for (size_t n = 0; n < N; ++n) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
          if (!IMPLICIT_TRANSPOSE) {
            z.at(b, m, n) += x.at(b, k, n) * y.at(b, m, k);
          } else {
            z.at(b, m, n) += x.at(b, k, n) * y.at(b, k, m);
          }
        }
      }
    }
  }
}

// template <typename T, bool IMPLICIT_TRANSPOSE = false>
// void matmul_broadcast_second(Tensor<T> &x, Tensor<T> &y, Tensor<T> &z) {
//   assert(x.ndim() >= 2);
//   assert(y.ndim() >= 2);
//   assert(z.ndim() >= 2);
//   auto K = x.shape(0);
//   auto N = x.shape(1);
//   size_t M;
//   if (!IMPLICIT_TRANSPOSE) {
//     M = y.shape(0);
//     assert(K == y.shape(1));
//   } else {
//     M = y.shape(1);
//     assert(K == y.shape(0));
//   }
//   assert(z.shape(0) == M);
//   assert(z.shape(1) == N);

//   auto B = x.size() / N / K;
//   assert(y.size() / M / K == 1);
//   assert(z.size() / M / N == B);
//   auto x_shape = x.shape();
//   auto y_shape = y.shape();
//   auto z_shape = z.shape();

//   x.view({K, N, B});
//   if (!IMPLICIT_TRANSPOSE) {
//     y.view({M, K, 1});
//   } else {
//     y.view({K, M, 1});
//   }
//   z.view({M, N, B});

//   z.zeros();
//   for (size_t b = 0; b < B; ++b) {
//     for (size_t n = 0; n < N; ++n) {
//       for (size_t m = 0; m < M; ++m) {
//         for (size_t k = 0; k < K; ++k) {
//           if (!IMPLICIT_TRANSPOSE) {
//             z.at(m, n, b) += x.at(k, n, b) * y.at(m, k);
//           } else {
//             z.at(m, n, b) += x.at(k, n, b) * y.at(k, m);
//           }
//         }
//       }
//     }
//   }

//   x.view(x_shape);
//   y.view(y_shape);
//   z.view(z_shape);
// }

} // namespace func