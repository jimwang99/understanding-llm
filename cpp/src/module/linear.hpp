#pragma once

#include "module.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void linear_out(const Tensor<T> &activation, const Tensor<T> &weight,
                const Tensor<T> &bias, bool has_bias, Tensor<T> &output) {
  assert(activation.ndim() >= 2);
  assert(weight.ndim() == 2);
  assert(bias.ndim() == 1);

  const auto size_in = activation.shape(0);
  assert(size_in == weight.shape(0));
  const auto size_out = weight.shape(1);
  assert(size_out == bias.shape(0));

  auto shape = activation.shape();
  shape[0] = size_out;

  output.reshape(shape);
  output.Zeros();

  auto batch = activation.size() / size_in;

  for (size_t k = 0; k < batch; ++k) {
    for (size_t j = 0; j < size_out; ++j) {
      for (size_t i = 0; i < size_in; ++i) {
        output.at(k, j) += weight.at(j, i) * activation.at(k, i);
      }
      if (has_bias) {
        output.at(k, j) += bias.at(j);
      }
    }
  }
}

}  // namespace func

namespace module {

template <typename T>
class Linear : public virtual Module<T> {
 protected:
  size_t len_in_;  /*! input sequence length */
  size_t len_out_; /*! output sequence length */
  bool has_bias_;  /*! whether to use bias */

  Tensor<T> weight_;
  Tensor<T> bias_;

  Tensor<T> &input_;
  Tensor<T> &output_;

 public:
  Linear(const std::string name, const size_t len_in, const size_t len_out,
         const bool has_bias, Tensor<T> &input, Tensor<T> &output)
      : Module<T>(name),
        len_in_(len_in),
        len_out_(len_out),
        has_bias_(has_bias),
        weight_({len_in, len_out}, "weight"),
        bias_({len_out}, "bias"),
        input_(input),
        output_(output) {
    add_param("len_in", len_in_);
    add_param("len_out", len_out_);
    add_param("has_bias", has_bias_);
    add_weight(weight_);
    if (has_bias_) {
      add_weight(bias_);
    }
    add_input(input_);
    add_outpu(output_);

    this->logger_ = setup_logger("linear");
    DEBUG("New module of Linear:\n{}", this->str());
  }

  void forward() override {
    func::linear_out(input_, weight_, bias_, has_bias_, output_);
  }
};

}  // namespace module