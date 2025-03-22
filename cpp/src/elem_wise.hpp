#pragma once

#include "logger.hpp"
#include "tensor.hpp"

namespace func {

static const std::string logger_name = "elem_wise.hpp";

template <typename T, char OP>
void _op_inline(Tensor<T> &x, const Tensor<T> &y) {
  auto logger = get_logger(logger_name);
  logger->trace("arg0 {}", x.str());
  logger->trace("arg1 {}", y.str());

  ASSERT(x.shape() == y.shape(), "x={} y={}", x.shape(), y.shape());

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
      x.at(i) = x.at(i) / y.at(i);
    }
  }
  logger->trace("result {}", x.str());
}

template <typename T>
void add_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '+'>(x, y);
}
template <typename T>
void sub_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '-'>(x, y);
}
template <typename T>
void mul_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '*'>(x, y);
}
template <typename T>
void div_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '/'>(x, y);
}

template <typename T, char OP>
void _op_scalar_inline(Tensor<T> &x, const T scalar) {
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
}

template <typename T>
void add_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '+'>(x, scalar);
}
template <typename T>
void sub_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '-'>(x, scalar);
}
template <typename T>
void mul_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '*'>(x, scalar);
}
template <typename T>
void div_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '/'>(x, scalar);
}

template <typename T, char OP>
void _op_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  ASSERT(x.ndim() == 2, x.shape());
  ASSERT(y.ndim() == 1, y.shape());
  ASSERT(x.shape(0) == y.shape(0), "x={} y={}", x.shape(), y.shape())

  for (size_t i = 0; i < x.shape(1); ++i) {
    for (size_t j = 0; j < x.shape(0); ++j) {
      if constexpr (OP == '+') {
        x.at(j, i) = x.at(j, i) + y.at(j);
      }
      if constexpr (OP == '-') {
        x.at(j, i) = x.at(j, i) - y.at(j);
      }
      if constexpr (OP == '*') {
        x.at(j, i) = x.at(j, i) * y.at(j);
      }
      if constexpr (OP == '/') {
        x.at(j, i) = x.at(j, i) / y.at(j);
      }
    }
  }
}

template <typename T>
void add_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_broadcast_inline<T, '+'>(x, y);
}
template <typename T>
void sub_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_broadcast_inline<T, '-'>(x, y);
}
template <typename T>
void mul_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_broadcast_inline<T, '*'>(x, y);
}
template <typename T>
void div_broadcast_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_broadcast_inline<T, '/'>(x, y);
}

}  // namespace func