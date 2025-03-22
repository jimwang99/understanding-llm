#pragma once

#include "logger.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void tranpose_1_2_inline(Tensor<T> &x) {
  ASSERT(x.ndim() >= 3, x.shape());
  auto x_shape = x.shape();

  auto D0 = x_shape[0];
  auto D1 = x_shape[1];
  auto D2 = x_shape[2];
  auto D3 = x.size() / (D0 * D1 * D2);
  x.view({D0, D1, D2, D3});

  auto x_stride = x.stride();
  auto S0 = x_stride[0];
  auto S1 = x_stride[1];
  auto S2 = x_stride[2];
  auto S3 = x_stride[3];

  Tensor<T> buf({D0});

  auto offset3, offset2, offset1;
  auto base_addr = reinterpret_cast<size_t>(x.data());
  auto size = D0 * sizeof(T);
  for (auto i3 = 0; i3 < D3; ++i3) {
    offset3 = i3 * S3 + base_addr;
    for (auto i2 = 0; i2 < D2; ++i2) {
      offset2 = i2 * S2 + offset3;
      offset1 = i2 * S1 + offset3;
      for (auto i1 = 0; i1 < D1; ++i1) {
        auto swap0 = offset2 + i1 * S1;
        auto swap1 = offset1 + i1 * S2;
        memcpy(buf.data(), reinterpret_cast<T *>(swap0), size);
        memcpy(reinterpret_cast<T *>(swap0), reinterpret_cast<T *>(swap1),
               size);
        memcpy(reinterpret_cast<T *>(swap1), buf.data(), size);
      }
    }
  }
}

}  // namespace func