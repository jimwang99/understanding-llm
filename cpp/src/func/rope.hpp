#pragma once

#include <cassert>

#include "tensor.hpp"

namespace func {

template <typename T>
void rope_inline(Tensor<T> &qk, const Tensor<T> &cos, const Tensor<T> &sin,
                 const size_t cache_size = 0) {
  ASSERT(qk.ndim() == 4, qk.shape());
  auto B = qk.shape(3);
  auto L = qk.shape(2);
  auto N = qk.shape(1);
  auto D = qk.shape(0);
  ASSERT(D % 2 == 0, D);
  if (cache_size > 0) {  // generation stage
    ASSERT(L == 1, "cache_size={} L={}", cache_size, L);
  }

  for (size_t b = 0; b < B; ++b) {
    for (size_t l = 0; l < L; ++l) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t d = 0; d < D; ++d) {
          auto i = qk.at(d, n, l, b);      // imaginary
          auto r = qk.at(d + 1, n, l, b);  // real
          qk.at(d, n, l, b) = r * cos_.at(d << 1, l + cache_size) +
                              i * sin_.at(d << 1, l + cache_size);
          qk.at(d + 1, n, l, b) = r * cos_.at(d << 1, l + cache_size) -
                                  i * sin_.at(d << 1, l + cache_size);
        }
      }
    }
  }
}

}  // namespace func