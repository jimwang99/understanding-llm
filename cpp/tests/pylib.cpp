#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "func/elem_wise.hpp"
#include "func/matmul.hpp"
#include "func/norm.hpp"
#include "func/softmax.hpp"
#include "module/modules.hpp"
#include "tensor.hpp"

#include "logger.hpp"

namespace py = pybind11;

logger_t logger;

template <typename T> py::array_t<T> tensor_to_numpy(const Tensor<T> &tensor) {
  py::array_t<T> array(tensor.size());
  std::memcpy(array.mutable_data(), tensor.data(), tensor.nbytes());
  return array.reshape(tensor.shape());
}

template <typename T> Tensor<T> numpy_to_tensor(const py::array_t<T> &array) {
  std::vector<size_t> shape(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    shape[i] = array.shape(i);
  }
  Tensor<T> tensor(shape);
  std::memcpy(tensor.mutable_data(), array.data(), tensor.nbytes());
  return tensor;
}

template <typename T, char OP>
py::array_t<T> py_op(const py::array_t<T> &x, const py::array_t<T> &y) {
  auto tx = numpy_to_tensor<T>(x);
  tx.set_name("X");
  logger->trace("arg0: {}", tx.str());
  auto ty = numpy_to_tensor<T>(y);
  ty.set_name("Y");
  logger->trace("arg1: {}", ty.str());
  func::_op_inline<T, OP>(tx, ty);
  logger->trace("result: {}", tx.str());
  return tensor_to_numpy<T>(tx);
}

template <typename T, char OP>
py::array_t<T> py_op_scalar(const py::array_t<T> &x, const T y) {
  auto tx = numpy_to_tensor<T>(x);
  func::_op_scalar_inline<T, OP>(tx, y);
  return tensor_to_numpy<T>(tx);
}

template <typename T>
py::array_t<T> py_rms_norm(const py::array_t<T> &x, const float eps) {
  auto tx = numpy_to_tensor<T>(x);
  func::rms_norm_inline(tx, eps);
  return tensor_to_numpy<T>(tx);
}

template <typename T> py::array_t<T> py_softmax(const py::array_t<T> &x) {
  auto tx = numpy_to_tensor<T>(x);
  func::softmax_inline(tx);
  return tensor_to_numpy<T>(tx);
}

template <typename T, bool IMPLICIT_TRANSPOSE = false>
py::array_t<T> py_matmul(const py::array_t<T> &x, const py::array_t<T> &y) {
  auto tx = numpy_to_tensor<T>(x);
  auto ty = numpy_to_tensor<T>(y);
  auto tz = Tensor<T>();
  if (!IMPLICIT_TRANSPOSE) {
    tz.reshape({ty.shape(0), tx.shape(1)});
  } else {
    tz.reshape({ty.shape(1), tx.shape(1)});
  }
  func::matmul_2d_out<T, IMPLICIT_TRANSPOSE>(tx, ty, tz);
  return tensor_to_numpy<T>(tx);
}

PYBIND11_MODULE(ullm, m) {
  logger = get_logger("pylib.cpp", "trace");
  get_logger("elem_wise.hpp", "trace");

  m.doc() = "Understanding LLM C++ Framework";

  m.def("add_fp32", &py_op<float, '+'>);
  m.def("sub_fp32", &py_op<float, '-'>);
  m.def("mul_fp32", &py_op<float, '*'>);
  m.def("div_fp32", &py_op<float, '/'>);

  m.def("add_scalar_fp32", &py_op_scalar<float, '+'>);
  m.def("sub_scalar_fp32", &py_op_scalar<float, '-'>);
  m.def("mul_scalar_fp32", &py_op_scalar<float, '*'>);
  m.def("div_scalar_fp32", &py_op_scalar<float, '/'>);

  m.def("matmul_fp32", &py_matmul<float>);
  m.def("matmulT_fp32", &py_matmul<float, true>);

  m.def("rms_norm_fp32", &py_rms_norm<float>);
  m.def("softmax_fp32", &py_softmax<float>);
}