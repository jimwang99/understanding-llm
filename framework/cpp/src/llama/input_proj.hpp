#pragma once

#include <memory>

#include "llama/hyper_param.hpp"
#include "module/linear.hpp"
#include "tensor.hpp"

namespace llama {

template <typename T>
class InputProj : public virtual Module<T> {
 protected:
  const HyperParam &hp_;

  Tensor<T> &emb_; /*! input token embeddings */
  Tensor<T> &q_;   /*! output query */
  Tensor<T> &k_;   /*! output key */
  Tensor<T> &v_;   /*! output value */

  Linear<T> m_proj_q_; /*! linear layer for query */
  Linear<T> m_proj_k_; /*! linear layer for key */
  Linear<T> m_proj_v_; /*! linear layer for value */

 public:
  InputProj(std::string name, const HyperParam &hp, const Tensor<T> &emb,
            Tensor<T> &q, Tensor<T> &k, Tensor<T> &v)
      : Module<T>(name),
        hp_(hp),
        emb_(emb),
        q_(q),
        k_(k),
        v_(v),
        m_proj_q_(name + ".proj_q", hp.D, hp.D, false, emb, q),
        m_proj_k_(name + ".proj_k", hp.D, hp.Dkv, false, emb, k),
        m_proj_v_(name + ".proj_v", hp.D, hp.Dkv, false, emb, v) {
    this->logger_ = setup_logger(name, "trace");
    add_input(emb_);
    add_output(q_);
    add_output(k_);
    add_output(v_);
    add_submodule(m_proj_q_);
    add_submodule(m_proj_k_);
    add_submodule(m_proj_v_);
    INFO("New module of InputProj:\n{}", this->str());
  }
};  // class InputProj

}  // namespace llama