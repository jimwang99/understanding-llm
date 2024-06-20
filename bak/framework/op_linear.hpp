#pragma once

#include "tensor.hpp"

template <typename T>
void linear_1d_(const Tensor<T> &input, const Tensor<T> &weight,
                Tensor<T> &output) {
  auto [input_sz0, input_sz1, input_sz2, input_sz3] = input.size();
  auto [weight_sz0, weight_sz1, weight_sz2, weight_sz3] = weight.size();
  auto [output_sz0, output_sz1, output_sz2, output_sz3] = output.size();

  check_output_size_linear(input_sz0, input_sz1, input_sz2, input_sz3,
                           weight_sz0, weight_sz1, weight_sz2, weight_sz3,
                           output_sz0, output_sz1, output_sz2, output_sz3);

  auto sz_012 = input_sz0 * input_sz1 * input_sz2;

  input.view(1, 1, sz_012, input_sz3);
  output.view(1, 1, sz_012, output_sz3);

  for (auto i0 = 0; i0 < sz_012; ++i0) {
    auto ofs_input = i0 * input_sz3;
    auto ofs_output = i0 * output_sz3;
    for (auto i1 = 0; i1 < output_sz3; ++i1) {
      auto ofs_weight = i1 * input_sz3;
      T v = 0;
      for (auto i2 = 0; i2 < input_sz3; ++i2) {
        v += input.at(ofs_input + i2) * weight.at(ofs_weight + i2);
      }
      output.at(ofs_output + i1) = v;
    }
  }

  input.view(input_sz0, input_sz1, input_sz2, input_sz3);
  output.view(output_sz0, output_sz1, output_sz2, output_sz3);
}