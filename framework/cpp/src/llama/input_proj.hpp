#pragma once

#include "llama/hyper_param.hpp"
#include "module/linear.hpp"
#include "tensor.hpp"
#include <memory>

namespace llama {

using namespace module;

template <typename T> class InputProj : public virtual Module<T> {
protected:
  const HyperParam &hp_;

  std::shared_ptr<Linear<T>> m_proj_q_; /*! linear layer for query */
  std::shared_ptr<Linear<T>> m_proj_k_; /*! linear layer for key */
  std::shared_ptr<Linear<T>> m_proj_v_; /*! linear layer for value */

  std::shared_ptr<const Tensor<T>> emb_; /*! input token embeddings */
  std::shared_ptr<Tensor<T>> q_;         /*! output query */
  std::shared_ptr<Tensor<T>> k_;         /*! output key */
  std::shared_ptr<Tensor<T>> v_;         /*! output value */

public:
  InputProj(std::string name, const HyperParam &hp, const Tensor<T> &emb,
            Tensor<T> &q, Tensor<T> &k, Tensor<T> &v)
      : Module<T>(name), hp_(hp),
        m_proj_q_(std::make_shared<Linear<T>>(name + ".proj_q", hp.D, hp.D,
                                              false, emb, q)),
        m_proj_k_(std::make_shared<Linear<T>>(name + ".proj_k", hp.D, hp.Dkv,
                                              false, emb, k)),
        m_proj_v_(std::make_shared<Linear<T>>(name + ".proj_v", hp.D, hp.Dkv,
                                              false, emb, v)),
        emb_(std::make_shared<const Tensor<T>>(emb)),
        q_(std::make_shared<Tensor<T>>(q)), k_(std::make_shared<Tensor<T>>(k)),
        v_(std::make_shared<Tensor<T>>(v)) {
    this->submodules_.push_back(m_proj_q_);
    this->submodules_.push_back(m_proj_k_);
    this->submodules_.push_back(m_proj_v_);
    this->inputs_.push_back(emb_);
    this->outputs_.push_back(q_);
    this->outputs_.push_back(k_);
    this->outputs_.push_back(v_);
    spdlog::info("New module of InputProj");
    spdlog::info("{}", this->str());
  }
}; // class InputProj

} // namespace llama