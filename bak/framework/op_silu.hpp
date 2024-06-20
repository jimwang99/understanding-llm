#include <cmath>

void silu_float(const Tensor<float>& input, Tensor<T>& output) {
  for (auto i0 = 0; i0 < input.numel(); ++i0) {
    output.at(i0) = 1.0 / (1.0 + exp(-1.0 * input.at(i0)));
  }
}