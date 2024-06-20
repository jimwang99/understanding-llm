#include <cassert>
#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

int64_t _conv2d_kernel(int64_t *ptr_input, int64_t *ptr_kernel, py::ssize_t Ci,
                       py::ssize_t Hk, py::ssize_t Wk) {
  int64_t result = 0;
  for (auto cii = 0; cii < Ci; ++cii) {
    result += ptr_input[cii] * ptr_kernel[cii];
  }
  return result;
}

py::array_t<int64_t>
conv2d(py::array_t<int64_t, py::array::forcecast> tensor_input,
       py::array_t<int64_t, py::array::forcecast> tensor_kernel) {
  // check inputs
  if (tensor_input.ndim() != 4 || tensor_kernel.ndim() != 4) {
    throw std::runtime_error("Number of dimensions of both inputs must be 4");
  }

  // get inputs' shapes
  auto N = tensor_input.shape(0);
  auto Hi = tensor_input.shape(1);
  auto Wi = tensor_input.shape(2);
  auto Ci = tensor_input.shape(3);
  spdlog::debug("tensor_input.shape=({},{},{},{})", N, Hi, Wi, Ci);

  auto Co = tensor_kernel.shape(0);
  auto Hk = tensor_kernel.shape(1);
  auto Wk = tensor_kernel.shape(2);
  auto Ck = tensor_kernel.shape(3);
  spdlog::debug("tensor_kernel.shape=({},{},{},{})", Co, Hk, Wk, Ck);

  if (Ci != Ck) {
    throw std::runtime_error("Input channel # doesn't match kernel channel #");
  }

  py::ssize_t Ph = 0, Pw = 0; // padding
  py::ssize_t Dh = 1, Dw = 1; // dilation
  py::ssize_t Sh = 1, Sw = 1; // stride
  spdlog::debug("padding=({},{}) dilation=({},{}), stride=({},{})", Ph, Pw, Dh,
                Dw, Sh, Sw);

  // calculate output's shape
  auto Ho = py::ssize_t(floor((Hi + 2 * Ph - Dh * (Hk - 1) - 1) / Sh + 1));
  auto Wo = py::ssize_t(floor((Wi + 2 * Pw - Dw * (Wk - 1) - 1) / Sw + 1));

  // construct output tensor
  auto tensor_output = py::array_t<int64_t>(
      N * Ho * Wo * Co); // allocate the buffer for output tensor
  tensor_output = tensor_output.reshape({N, Ho, Wo, Co}); // from 1D to 4D
  spdlog::debug("tensor_output.shape=({},{},{},{})", N, Ho, Wo, Co);

  // get pointer
  auto buf_input = tensor_input.request();
  auto buf_kernel = tensor_kernel.request();
  auto buf_output = tensor_output.request();

  int64_t *ptr_input = static_cast<int64_t *>(buf_input.ptr);
  int64_t *ptr_kernel = static_cast<int64_t *>(buf_kernel.ptr);
  int64_t *ptr_output = static_cast<int64_t *>(buf_output.ptr);

  for (auto ni = 0; ni < N; ++ni) {           // N
    for (auto coi = 0; coi < Co; ++coi) {     // C of output
      for (auto hoi = 0; hoi < Ho; ++hoi) {   // H of output
        for (auto woi = 0; woi < Wo; ++woi) { // W of output
          // calculate index of target elem in output
          auto io = ni * Ho * Wo * Co + hoi * Wo * Co + woi * Co + coi;

          // calculate index of first elem in input tensor
          auto hii0 = hoi, wii0 = woi;
          assert(hii0 < Hi);
          assert(wii0 < Wi);
          auto ii0 = ni * Hi * Wi * Ci + hii0 * Wi * Ci + wii0 * Ci;
          spdlog::trace("io={} ({},{},{},{}) . ii0={} ({},{},{},0)", io, ni,
                        hoi, woi, coi, ii0, ni, hii0, wii0);

          // MAC
          int64_t mac = 0;
          for (auto hki = 0; hki < Hk; ++hki) {     // H of kernel
            for (auto wki = 0; wki < Wk; ++wki) {   // W of kernel
              for (auto cki = 0; cki < Ck; ++cki) { // C of kernel
                // calculate index of element in kernel tensor
                auto ki = coi * Hk * Wk * Ck + hki * Wk * Ck + wki * Ck + cki;
                // calculate index of element in input tensor
                auto hii = hii0 + hki;
                auto wii = wii0 + wki;
                auto cii = cki;
                auto ii = ii0 + hki * Wi * Ci + wki * Ci + cii;
                spdlog::trace("  ki={} ({},{},{},{}) .  ii={} ({},{},{},{})",
                              ki, coi, hki, wki, cki, ii, ni, hii, wii, cii);

                mac += ptr_input[ii] * ptr_kernel[ki];
              }
            }
          }

          // write to output tensor
          ptr_output[io] = mac;
        }
      }
    }
  }

  return tensor_output;
}

// wrap as Python module
PYBIND11_MODULE(conv2d, m) {
  // setup logger
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::debug);

  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
      "/tmp/conv2d.log", true);
  file_sink->set_level(spdlog::level::trace);

  auto logger = std::make_shared<spdlog::logger>(
      "conv2d_dot_cpp", spdlog::sinks_init_list({console_sink, file_sink}));
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::trace);

  m.doc() = "conv2d implementation in C++";
  m.def("conv2d", &conv2d, "basic conv2d operation");
}
