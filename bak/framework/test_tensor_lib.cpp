#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.hpp"
#include "tensor_helper.hpp"

namespace py = pybind11;

py::array_t<float> test_tensor4_init_float32(py::array_t<float> a) {
  auto b = convert_ndarray_to_tensor4<float>(a);
  return convert_tensor4_to_ndarray(b);
}

py::array_t<int32_t> test_tensor4_init_int32(py::array_t<int32_t> a) {
  auto b = convert_ndarray_to_tensor4<int32_t>(a);
  return convert_tensor4_to_ndarray(b);
}

PYBIND11_MODULE(test_tensor_lib, m) {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::debug);

  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "/tmp/understanding_llm/acc/test_tensor_lib.log", true);
  file_sink->set_level(spdlog::level::trace);

  auto logger = std::make_shared<spdlog::logger>(
      "test_tensor_lib", spdlog::sinks_init_list({console_sink, file_sink}));
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::trace);

  m.doc() = "Library for testing tensor.hpp";
  m.def("test_tensor4_init_float32", &test_tensor4_init_float32,
        "Test Tensor4<float> initialization with float32 numpy array");
  m.def("test_tensor4_init_int32", &test_tensor4_init_int32,
        "Test Tensor4<int32> initialization with int32 numpy array");
}