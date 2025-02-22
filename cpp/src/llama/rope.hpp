#pragma once

#include "tensor.hpp"
#include <cassert>

template <typename T, size_t Lm> class RoPE {
public:
  RoPE() { init(); }

  Tensor<T> forward(const Tensor<T> &x, const size_t Lc = 0) {
    assert(x.ndim() == 4);
    auto B = x.shape(3);
    auto L = x.shape(2);
    auto N = x.shape(1);
    auto D = x.shape(0);
    assert(D % 2 == 0);
    if (Lc > 0) { // generation stage
      assert(L == 1);
    }

    Tensor<T> y(x.shape());
    for (size_t b = 0; b < B; ++b) {
      for (size_t l = 0; l < L; ++l) {
        for (size_t n = 0; n < N; ++n) {
          for (size_t d = 0; d < D; ++d) {
            auto i = x.at(d, n, l, b);
            auto r = x.at(d + 1, n, l, b);
            y.at(d, n, l, b) =
                r * cos_.at(d << 1, l + Lc) + i * sin_.at(d << 1, l + Lc);
            y.at(d + 1, n, l, b) =
                r * cos_.at(d << 1, l + Lc) - i * sin_.at(d << 1, l + Lc);
          }
        }
      }
    }
    return y;
  }

private:
  Tensor<T> cos_;
  Tensor<T> sin_;

  void init() {}
};