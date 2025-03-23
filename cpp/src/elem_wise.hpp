#pragma once

#include "logger.hpp"
#include "tensor.hpp"
#include <functional>

namespace func {

namespace detail {

template <typename T, char OP>
void _op_inline(Tensor<T> &x, const Tensor<T> &y) {
  auto logger = setup_logger();
  ASSERT(x.shape() == y.shape(), "shape mismatch: x={} y={}", x.shape(),
         y.shape());

  TRACE("tensor-x {}", x.str());
  TRACE("tensor-y {}", y.str());

  for (size_t i = 0; i < x.size(); ++i) {
    if constexpr (OP == '+') {
      x.at(i) = x.at(i) + y.at(i);
    }
    if constexpr (OP == '-') {
      x.at(i) = x.at(i) - y.at(i);
    }
    if constexpr (OP == '*') {
      x.at(i) = x.at(i) * y.at(i);
    }
    if constexpr (OP == '/') {
      ASSERT(y.at(i) != 0, "division by zero");
      x.at(i) = x.at(i) / y.at(i);
    }
  }

  TRACE("result {}", x.str());
} // _op_inline
template <typename T, char OP> void _op_scalar_inline(Tensor<T> &x, T scalar) {
  auto logger = setup_logger();
  TRACE("tensor {}", x.str());
  TRACE("scalar {}", scalar);

  if constexpr (OP == '/') {
    ASSERT(scalar != 0, "division by zero");
  }

  for (size_t i = 0; i < x.size(); ++i) {
    if constexpr (OP == '+') {
      x.at(i) = x.at(i) + scalar;
    }
    if constexpr (OP == '-') {
      x.at(i) = x.at(i) - scalar;
    }
    if constexpr (OP == '*') {
      x.at(i) = x.at(i) * scalar;
    }
    if constexpr (OP == '/') {
      x.at(i) = x.at(i) / scalar;
    }
  }

  TRACE("result {}", x.str());
} // _op_scalar_inline

template <typename T, char OP>
void _op_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  auto logger = setup_logger();
  ASSERT(x.ndim() == 2, "shape mismatch: x must be 2D, got shape={}",
         x.shape());
  ASSERT(y.ndim() == 1, "shape mismatch: y must be 1D, got shape={}",
         y.shape());
  ASSERT(x.shape(0) == y.shape(0),
         "shape mismatch: x inner size = {} must match y size = {}", x.shape(0),
         y.shape(0));

  TRACE("tensor-x {}", x.str());
  TRACE("tensor-y {}", y.str());

  for (size_t j = 0; j < x.shape(1); ++j) {
    for (size_t i = 0; i < x.shape(0); ++i) {
      if constexpr (OP == '+') {
        x.at(i, j) = x.at(i, j) + y.at(i);
      }
      if constexpr (OP == '-') {
        x.at(i, j) = x.at(i, j) - y.at(i);
      }
      if constexpr (OP == '*') {
        x.at(i, j) = x.at(i, j) * y.at(i);
      }
      if constexpr (OP == '/') {
        ASSERT(y.at(i) != 0, "division by zero");
        x.at(i, j) = x.at(i, j) / y.at(i);
      }
    }
  }

  TRACE("result {}", x.str());
} // _op_broadcast_inline

} // namespace detail

template <typename T> void add_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_inline<T, '+'>(x, y);
}
template <typename T> void sub_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_inline<T, '-'>(x, y);
}
template <typename T> void mul_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_inline<T, '*'>(x, y);
}
template <typename T> void div_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_inline<T, '/'>(x, y);
}

template <typename T> void add_scalar_inline(Tensor<T> &x, T scalar) {
  detail::_op_scalar_inline<T, '+'>(x, scalar);
}
template <typename T> void sub_scalar_inline(Tensor<T> &x, T scalar) {
  detail::_op_scalar_inline<T, '-'>(x, scalar);
}
template <typename T> void mul_scalar_inline(Tensor<T> &x, T scalar) {
  detail::_op_scalar_inline<T, '*'>(x, scalar);
}
template <typename T> void div_scalar_inline(Tensor<T> &x, T scalar) {
  detail::_op_scalar_inline<T, '/'>(x, scalar);
}

template <typename T>
void add_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_broadcast_inline<T, '+'>(x, y);
}
template <typename T>
void sub_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_broadcast_inline<T, '-'>(x, y);
}
template <typename T>
void mul_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_broadcast_inline<T, '*'>(x, y);
}
template <typename T>
void div_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  detail::_op_broadcast_inline<T, '/'>(x, y);
}

} // namespace func