#pragma once

#include "input_norm.hpp"
#include "input_proj.hpp"
#include "module.hpp"

namespace llama {

template <typename T>
class Layer : public virtual Module<T> {
 private:
  const HyperParam &hp_;

  Tensor<T> &emb_; /*! input and output of Layer*/

  Tensor<T> emb_norm_;
  Tensor<T> q_;
  Tensor<T> k_;
  Tensor<T> v_;

  InputNorm<T> m_input_norm_;
  InputProj<T> m_input_proj_;

 public:
  Layer(const std::string name, const HyperParam &hp, Tensor<T> &emb)
      : Module<T>(name),
        hp_(hp),
        emb_(emb),
        emb_norm_({hp_.B, hp_.Lm, hp_.D}, "emb_norm"),
        m_input_norm_(name + ".input_norm", hp_, emb_, emb_norm_),
        m_input_proj_(name + ".input_proj", hp_, embn_, )

  {
    this->logger_ = setup_logger(name, "trace");
    add_inout(emb_);
    INFO("New module of Layer:\n{}", this->str());
  }

  void forward() {
    // TODO
  }
};

}  // namespace llama