#pragma once

#include "func/norm.hpp"
#include "module.hpp"

namespace llama {

template <typename T>
class InputNorm : public virtual Module<T> {
 private:
  const HyperParam &hp_;

  Tensor<T> &input_;
  Tensor<T> &output_;

 public:
  InputNorm(const std::string name, const HyperParam &hp, Tensor<T> &input,
            Tensor<T> &output)
      : Module<T>(name), hp_(hp), input_(input_), output_(output) {
    this->logger_ = setup_logger("input_norm");
    add_input(input_);
    add_output(output_);
    DEBUG("New module of InputNorm:\n{}", this->str())
  }

  void forward() { func::rms_norm_inline(emb_, hp_.Eps); }
};

}  // namespace llama