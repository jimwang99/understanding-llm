#pragma once

// #include "input_proj.hpp"
#include "module.hpp"

namespace llama {

template <typename T>
class Layer : public virtual Module<T> {
 private:
  HyperParam hp_;

  Tensor<T> &emb_; /*! input and output of Layer*/

 public:
  Layer(std::string name, const HyperParam &hp, Tensor<T> &emb_)
      : Module<T>(name), hp_(hp), emb_(emb) {
    add_inout(emb_);
    logger_ = setup_logger("layer");
    DEBUG("New module of Layer:\n{}", this->str());
  }
};

}  // namespace llama