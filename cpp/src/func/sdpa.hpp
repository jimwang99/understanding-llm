#pragma once

#include <cmath>

#include "elem_wise.hpp"
#include "logger.hpp"
#include "matmul.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void init_mask(Tensor<T> &mask, const size_t max_seq_len) {
  mask.reshape({1, 1, max_seq_len, max_seq_len});
  mask.zeros();
}

template <typename T>
void sdpa_out(Tensor<T> &q, Tensor<T> &k, Tensor<T> &v, Tensor<T> &attn,
              const Tensor<T> &mask) {
  ASSERT(q.ndim() == 4, q.shape());
  ASSERT(k.ndim() == 4, k.shape());
  ASSERT(v.ndim() == 4, v.shape());

  ASSERT(q.shape(0) == k.shape(0), "Q={} K={}", q.shape(), k.shape());
  ASSERT(q.shape(2) == k.shape(2), "Q={} K={}", q.shape(), k.shape());
  ASSERT(q.shape(3) == k.shape(3), "Q={} K={}", q.shape(), k.shape());

  ASSERT(k.shape() == v.shape(), "K={} V={}", k.shape(), v.shape());

  auto B = q.shape(3);
  auto N = q.shape(2);
  auto D = q.shape(0);
  auto Lq = q.shape(1);
  auto Lkv = k.shape(1);

  // Q⋅Kᵀ
  q.view({B * N, Lq, D});
  k.view({B * N, Lkv, D});
  attn.view({B * N, Lq, Lkv});
  matmul_2d_out<T, true>(q, k, attn);

  // mask(Q⋅Kᵀ)
  for (size_t b = 0; b < B * N; ++b) {
    for (size_t i = 0; i < Lq; ++i) {
      for (size_t j = 0; j < Lkv; ++j) {
        attn.at(b, i, j) =
            attn.at(b, i, j) + mask.at(b, i, j);  // TODO: for generation stage
      }
    }
  }

  // mask(Q⋅Kᵀ/√D)
  // TODO: potential optimization: skip inf number
  div_scalar_inline<T>(attn, std::sqrt(D));

  // softmax(mask(Q⋅Kᵀ/√D))
  softmax_inline(m, true);

  // V⋅softmax(mask(Q⋅Kᵀ/√D))
  v.view({B * N, Lkv, D});
  matmul_2d_out<T, false>(sm, v, q);

  q.view({B, N, Lq, D});
  k.view({B, N, Lkv, D});
  v.view({B, N, Lkv, D});
  attn.view({B, N, Lq, Lkv});
}

}  // namespace func