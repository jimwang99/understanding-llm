#pragma once

#include "tensor.hpp"
#include <cassert>

#include "logger.hpp"

namespace func {

static const std::string logger_name = "elem_wise.hpp";

template <typename T, char OP>
void _op_inline(Tensor<T> &x, const Tensor<T> &y) {
  auto logger = get_logger(logger_name);
  logger->trace("arg0 {}", x.str());
  logger->trace("arg1 {}", y.str());

  assert(x.shape() == y.shape());

  for (size_t i = 0; i < x.size(); ++i) {
    if (OP == '+') {
      x.at(i) = x.at(i) + y.at(i);
    }
    if (OP == '-') {
      x.at(i) = x.at(i) - y.at(i);
    }
    if (OP == '*') {
      x.at(i) = x.at(i) * y.at(i);
    }
    if (OP == '/') {
      x.at(i) = x.at(i) / y.at(i);
    }
  }
  logger->trace("result {}", x.str());
}

template <typename T> void add_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '+'>(x, y);
}
template <typename T> void sub_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '-'>(x, y);
}
template <typename T> void mul_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '*'>(x, y);
}
template <typename T> void div_inline(Tensor<T> &x, const Tensor<T> &y) {
  _op_inline<T, '/'>(x, y);
}

template <typename T, char OP>
void _op_scalar_inline(Tensor<T> &x, const T scalar) {
  for (size_t i = 0; i < x.size(); ++i) {
    if (OP == '+') {
      x.at(i) = x.at(i) + scalar;
    }
    if (OP == '-') {
      x.at(i) = x.at(i) - scalar;
    }
    if (OP == '*') {
      x.at(i) = x.at(i) * scalar;
    }
    if (OP == '/') {
      x.at(i) = x.at(i) / scalar;
    }
  }
}

template <typename T> void add_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '+'>(x, scalar);
}
template <typename T> void sub_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '-'>(x, scalar);
}
template <typename T> void mul_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '*'>(x, scalar);
}
template <typename T> void div_scalar_inline(Tensor<T> &x, const T scalar) {
  _op_scalar_inline<T, '/'>(x, scalar);
}

// template <typename T, int NDIM, char OP>
// void InlineOperationBroadcast(Tensor<T> &x, Tensor<T> &y) {
//   assert(x.ndim() == y.ndim());
//   assert(x.size() >= y.size());

//   size_t size_inner = 1;
//   for (int i = 0; i < NDIM; ++i) {
//     assert(x.shape(i) == y.shape(i));
//     size_inner *= x.shape(i);
//   }
//   assert(size_inner == y.size());
//   size_t size_outer = x.size() / size_inner;

//   const auto x_shape = x.shape();
//   const auto y_shape = y.shape();

//   x.view({size_outer, size_inner});
//   y.view({size_inner});

//   for (size_t i = 0; i < size_outer; ++i) {
//     for (size_t j = 0; j < size_inner; ++j) {
//       if (OP == '+') {
//         x.at(j, i) = x.at(j, i) + y.at(j);
//       }
//       if (OP == '-') {
//         x.at(j, i) = x.at(j, i) - y.at(j);
//       }
//       if (OP == '*') {
//         x.at(j, i) = x.at(j, i) * y.at(j);
//       }
//       if (OP == '/') {
//         x.at(j, i) = x.at(j, i) / y.at(j);
//       }
//     }
//   }
//   x.view(x_shape);
//   y.view(y_shape);
// }

// template <typename T, int NDIM>
// void InlineAddBroadcast(Tensor<T> &x, Tensor<T> &y) {
//   InlineOperationBroadcast<T, NDIM, '+'>(x, y);
// }
// template <typename T, int NDIM>
// void InlineSubBroadcast(Tensor<T> &x, Tensor<T> &y) {
//   InlineOperationBroadcast<T, NDIM, '-'>(x, y);
// }
// template <typename T, int NDIM>
// void InlineMulBroadcast(Tensor<T> &x, Tensor<T> &y) {
//   InlineOperationBroadcast<T, NDIM, '*'>(x, y);
// }
// template <typename T, int NDIM>
// void InlineDivBroadcast(Tensor<T> &x, Tensor<T> &y) {
//   InlineOperationBroadcast<T, NDIM, '/'>(x, y);
// }

} // namespace func