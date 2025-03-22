#pragma once

#include "activation.hpp"
#include "module.hpp"
#include "tensor.hpp"

namespace func {

template <typename T, bool has_bias = false, bool has_addend = false,
          bool has_silu = false>
void linear_out(
    const Tensor<T> &activation, const Tensor<T> &weight, Tensor<T> &output,
    const std::optional<std::reference_wrapper<Tensor<T>>> bias = std::nullopt,
    const std::optional<std::reference_wrapper<Tensor<T>>> addend =
        std::nullopt) {
  assert(activation.ndim() >= 2);
  assert(weight.ndim() == 2);

  if constexpr (has_bias) {
    assert(bias.ndim() == 1);
    assert(bias.shape(0) == output.shape(0));
  }
  if constexpr (has_addend) {
    assert(addend.shape() == output.shape());
  }
  const Tensor<T> &b = bias.value_or(activation);
  const Tensor<T> &a = addend.value_or(activation);

  const auto size_in = activation.shape(0);
  assert(size_in == weight.shape(0));
  const auto size_out = weight.shape(1);
  assert(size_out == bias.shape(0));

  auto shape = activation.shape();
  shape[0] = size_out;

  assert(output.shape() == shape);
  output.zeros();

  auto batch = activation.size() / size_in;

  for (size_t k = 0; k < batch; ++k) {
    for (size_t j = 0; j < size_out; ++j) {
      for (size_t i = 0; i < size_in; ++i) {
        if constexpr (has_addend) {
          output.at(k, j) += weight.at(j, i) * activation.at(k, i) + a.at(k, j);
        } else {
          output.at(k, j) += weight.at(j, i) * activation.at(k, i);
        }
      }
      if constexpr (has_bias) {
        output.at(k, j) += b.at(j);
      }
      if constexpr (has_silu) {
        output.at(k, j) = func::silu(output.at(k, j));
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

  Tensor<T> &input_;
  Tensor<T> &output_;

 public:
  Linear(const std::string name, const size_t len_in, const size_t len_out,
         Tensor<T> &input, Tensor<T> &output)
      : Module<T>(name),
        len_in_(len_in),
        len_out_(len_out),
        weight_({len_in, len_out}, "weight"),
        input_(input),
        output_(output),
        logger_(setup_logger("Linear")) {
    add_param("Li", len_in_);
    add_param("Lo", len_out_);
    add_weight(weight_);
    add_input(input_);
    add_outpu(output_);

    DEBUG("New module of Linear:\n{}", this->str());
  }

  void forward() override { func::linear_out<T>(input_, weight_, output_); }
};

template <typename T>
class LinearAdd : public virtual Linear<T> {
 protected:
  Tensor<T> &addend_;

 public:
  LinearAdd(const std::string name, const size_t len_in, const size_t len_out,
            Tensor<T> &input, Tensor<T> &addend, Tensor<T> &output)
      : Linear<T>(name, len_in, len_out, input, output), addend_(addend) {
    add_input(addend_);
    DEBUG("New module of LinearAdd:\n{}", this->str());
  }

  void forward() override {
    func::linear_out<T, has_addend = true>(input_, weight_, output_,
                                           addend = addend_);
  }
};

template <typename T>
class LinearSilu : public virtual Linear<T> {
 public:
  LinearSilu(const std::string name, const size_t len_in, const size_t len_out,
             Tensor<T> &input, Tensor<T> &output)
      : Linear<T>(name, len_in, len_out, input, output) {
    DEBUG("New module of LinearSilu:\n{}", this->str());
  }

  void forward() override {
    func::linear_out<T, has_silu = true>(input_, weight_, output_);
  }
};

}  // namespace module