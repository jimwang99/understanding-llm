#pragma once

#include "logger.hpp"
#include "tensor.hpp"

namespace func {

// transpose tensor's 1st and 2nd innermost dimension (count starts at 0)
template <typename T> void tranpose_1_2_inline(Tensor<T> &x) {
  ASSERT(x.ndim() >= 3, x.shape());
  auto shape = x.shape();
  auto D0 = x.shape(0);
  auto D1 = x.shape(1);
  auto D2 = x.shape(2);
  auto D3 = x.size() / (D0 * D1 * D2);

  if ((D1 == 1) || (D2 == 1)) {
    // no need to transpose
    return;
  }

  x.view({D0, D1, D2, D3});

  auto s1 = D0;
  auto s2old = D0 * D1;
  auto s2new = D0 * D2;
  auto s3 = D0 * D1 * D2;
  // DEBUG("D0={} D1={} D2={} D3={} s1={} s2={}/{} s3={}", D0, D1, D2, D3, s1,
  //       s2old, s2new, s3);

  Tensor<T> buf("buf", {D0});

  size_t offset3, offset2_old, offset2_new, offset1_old, offset1_new, addr_old,
      addr_new;
  size_t base_addr = reinterpret_cast<size_t>(x.data());
  size_t size = D0 * sizeof(T);
  for (size_t i3 = 0; i3 < D3; ++i3) {
    offset3 = i3 * s3;
    for (size_t i2 = 0; i2 < D2; ++i2) {
      offset2_old = i2 * s2old + offset3;
      offset2_new = i2 * s1 + offset3;
      for (size_t i1 = 0; i1 < D1; ++i1) {
        if (i1 == i2) {
          continue;
        }
        offset1_old = i1 * s1 + offset2_old;
        offset1_new = i1 * s2new + offset2_new;
        addr_old = offset1_old * sizeof(T) + base_addr;
        addr_new = offset1_new * sizeof(T) + base_addr;
        // TRACE("i1={} i2={} i3={} offset1_old={} offset1_new={} offset2_old={}
        // "
        //       "offset2_new={} offset3={} addr_old={} addr_new={}",
        //       i1, i2, i3, offset1_old, offset1_new, offset2_old, offset2_new,
        //       offset3, addr_old, addr_new);
        memcpy(buf.mutable_data(), reinterpret_cast<T *>(addr_old), size);
        memcpy(reinterpret_cast<T *>(addr_old), reinterpret_cast<T *>(addr_new),
               size);
        memcpy(reinterpret_cast<T *>(addr_new), buf.data(), size);
      }
    }
  }
  shape[1] = D2;
  shape[2] = D1;
  x.view(shape);
}

} // namespace func