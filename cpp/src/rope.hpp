#pragma once

#include <cassert>
#include <cmath>
#include <vector>

#include "matmul.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void init_cis(Tensor<T> &cos, Tensor<T> &sin, const size_t max_seq_len,
              const size_t head_dim, const T theta_base = 10000.0) {
  // theta = theta_base ^ (-2 * i / head_dim) for i in [0, head_dim/2)
  // abs_pos = i for i in [0, max_seq_len)
  // freqs = abs_pos * theta (its shape == {head_dim / 2 , max_seq_len})
  // cos = cos(freqs)
  // sin = sin(freqs)

  const size_t D2 = head_dim >> 1;
  const size_t L = max_seq_len;
  cos.reshape({D2, L});
  sin.reshape({D2, L});

  Tensor<T> theta("theta", {D2});
  for (size_t i = 0; i < D2; ++i) {
    theta.at(i) = std::pow(theta_base, (-2.0f * i) / D2);
  }

  Tensor<T> abs_pos("abs_pos", {L});
  abs_pos.linspace(0.0f, 1.0f);

  Tensor<T> freqs("freqs", {D2, L, 1});
  theta.view({D2, 1, 1});
  abs_pos.view({1, L, 1});
  matmul_2d_out(abs_pos, theta, freqs);
  freqs.view({D2, L});

  for (size_t i = 0; i < cos.size(); ++i) {
    cos.at(i) = std::cos(freqs.at(i));
    sin.at(i) = std::sin(freqs.at(i));
  }
}

template <typename T>
void rope_inline(Tensor<T> &qk, const Tensor<T> &cos, const Tensor<T> &sin,
                 const size_t cache_size = 0) {
  ASSERT(qk.ndim() == 4, qk.shape());
  ASSERT(qk.shape(0) % 2 == 0, qk.shape());
  auto B = qk.shape(3);
  auto L = qk.shape(2);
  auto N = qk.shape(1);
  auto D = qk.shape(0) >> 1;
  if (cache_size > 0) { // generation stage
    ASSERT(L == 1, "cache_size={} L={}", cache_size, L);
  }
  ASSERT(cos.ndim() == 2, cos.shape());
  ASSERT(sin.ndim() == 2, sin.shape());
  ASSERT(cos.shape() == sin.shape(), "cos.shape()={} sin.shape()={}",
         cos.shape(), sin.shape());
  ASSERT(cos.shape(0) == D, "cos.shape()={}", cos.shape());
  auto Lm = cos.shape(1);
  ASSERT(Lm >= L + cache_size, "Lm={} L={} cache_size={}", Lm, L, cache_size);

  for (size_t b = 0; b < B; ++b) {
    for (size_t l = 0; l < L; ++l) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t d = 0; d < D; ++d) {
          auto d2 = d << 1;
          auto i = qk.at(d2, n, l, b);     // imaginary
          auto r = qk.at(d2 + 1, n, l, b); // real
          qk.at(d2, n, l, b) =
              r * cos.at(d, l + cache_size) + i * sin.at(d, l + cache_size);
          qk.at(d2 + 1, n, l, b) =
              r * cos.at(d, l + cache_size) - i * sin.at(d, l + cache_size);
        }
      }
    }
  }
}

} // namespace func