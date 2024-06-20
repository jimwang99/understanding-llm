#pragma once

#include <pybind11/numpy.h>

#include "tensor.hpp"

namespace py = pybind11;

template <typename T> Tensor4<T> convert_ndarray_to_tensor4(py::array_t<T> a) {
  std::vector<size_t> shape;
  for (auto s : a.shape()) {
    shape.push_back(s);
  }
  Tensor4<T> z(shape);

  // copy data
  py::buffer_info buf_a;
  T *ptr_a = static_cast<T *>(buf_a.ptr);
  std::memcpy(z.data(), ptr_a, a.nbytes());

  return z;
}

template <typename T> py::array_t<T> convert_tensor4_to_ndarray(Tensor4<T> z) {
  py::array_t<T> a(z.numel());
  a = a.reshape(z.shape());
  py::buffer_info buf_a = a.request();
  T *ptr_a = static_cast<T *>(buf_a.ptr);

  // copy data
  std::memcpy(ptr_a, z.data(), z.nbytes());

  return a;
}