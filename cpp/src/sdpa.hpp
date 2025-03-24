#pragma once

#include <cmath>

#include "logger.hpp"
#include "matmul.hpp"
#include "softmax.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void init_causal_attn_mask(Tensor<T> &mask, const size_t max_seq_len) {
  mask.reshape({max_seq_len, max_seq_len});
  for (size_t i = 0; i < max_seq_len; ++i) {
    for (size_t j = 0; j < max_seq_len; ++j) {
      if (j > i) {
        mask.at(j, i) = -1.0f * std::numeric_limits<T>::infinity();
      } else {
        mask.at(j, i) = 0.0f;
      }
    }
  }
}

template <typename T>
void sdpa_out(Tensor<T> &q, Tensor<T> &k, Tensor<T> &v, Tensor<T> &attn,
              const Tensor<T> &mask) {
  auto logger_name = "SDPA";
  get_logger(logger_name, "info");

  NASSERT(q.ndim() == 4, q.shape());
  NASSERT(k.ndim() == 4, k.shape());
  NASSERT(v.ndim() == 4, v.shape());

  auto B = q.shape(3);
  auto N = q.shape(2);
  auto D = q.shape(0);
  auto Lq = q.shape(1);
  auto Lkv = k.shape(1);

  NASSERT(k.shape() == v.shape(), "K={} V={}", k.shape(), v.shape());

  NASSERT(k.shape(0) == D, "Q={} K={}", q.shape(), k.shape());
  NASSERT(k.shape(2) == N, "Q={} K={}", q.shape(), k.shape());
  NASSERT(k.shape(3) == B, "Q={} K={}", q.shape(), k.shape());

  NASSERT(attn.ndim() == 4, attn.shape());
  NASSERT(attn.shape(0) == Lkv, "Q={} attn={}", q.shape(), attn.shape());
  NASSERT(attn.shape(1) == Lq, "Q={} attn={}", q.shape(), attn.shape());
  NASSERT(attn.shape(2) == N, "Q={} attn={}", q.shape(), attn.shape());
  NASSERT(attn.shape(3) == B, "Q={} attn={}", q.shape(), attn.shape());

  NASSERT(mask.ndim() == 2, mask.shape());
  NASSERT(mask.shape(0) == mask.shape(1), "mask={}", mask.shape());
  NASSERT(mask.shape(0) >= Lkv, "K={} mask={}", k.shape(), mask.shape());
  NASSERT(mask.shape(0) >= Lq, "Q={} mask={}", q.shape(), mask.shape());

  // attn = Q⋅Kᵀ
  q.view({D, Lq, B * N});
  k.view({D, Lkv, B * N});
  attn.view({Lkv, Lq, B * N});

  matmul_2d_out<T, true>(q, k, attn);

  NLOG_TRACE("====== Q⋅Kᵀ ======");
  NLOG_TRACE(q.str());
  NLOG_TRACE(k.str());
  NLOG_TRACE(attn.str());

  // attn = mask(Q⋅Kᵀ/√D)
  const T inv_sqrt_D = 1.0f / std::sqrt(D);
  for (size_t b = 0; b < B * N; ++b) {
    for (size_t i = 0; i < Lq; ++i) {
      for (size_t j = 0; j < Lkv; ++j) {
        attn.at(j, i, b) = attn.at(j, i, b) + mask.at(j, i);
        if (!std::isinf(attn.at(j, i, b))) {
          attn.at(j, i, b) *= inv_sqrt_D;
        }
      }
    }
  }

  NLOG_TRACE("====== attn = mask(attn/√D) ======");
  NLOG_TRACE(mask.str());
  NLOG_TRACE(attn.str());

  // attn = softmax(mask(Q⋅Kᵀ/√D))
  softmax_inline(attn, true);

  NLOG_TRACE("====== softmax ======");
  NLOG_TRACE(attn.str());

  // q = attn⋅V
  v.view({D, Lkv, B * N});
  matmul_2d_out<T, false>(attn, v, q);

  NLOG_TRACE("====== Q = attn⋅V ======");
  NLOG_TRACE(attn.str());
  NLOG_TRACE(v.str());
  NLOG_TRACE(q.str());

  q.view({D, Lq, N, B});
  k.view({D, Lkv, N, B});
  v.view({D, Lkv, N, B});
  attn.view({Lkv, Lq, N, B});
}

} // namespace func