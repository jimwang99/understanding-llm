/*
 *
 * llama
 *   embedding
 *   layer-<i>
 *     input-proj
 *     rope
 *     kv-cache
 *     sdpa
 *     attn-proj
 *     ffn
 *   output
 *   sampling
 *
 */

#include "hyper_param.hpp"
#include "layer.hpp"
#include "logger.hpp"
#include "module.hpp"
#include "module/embedding.hpp"
#include "module/linear.hpp"

namespace llama {

using namespace module;

template <typename T> class Llama final : public virtual Module<T> {
private:
  const HyperParam &hp_;

  Tensor<T> &token_;
  Tensor<T> emb_;
  Tensor<T> logit_;

  Embedding<T> m_embedding_;
  std::vector<Layer<T>> m_layers_;
  Linear<T> m_output_;

public:
  Llama(const std::string &name, const HyperParam &hp, Tensor<T> &token)
      : Module<T>(name), hp_(hp), token_(token),
        emb_({hp_.B, hp_.Lm, hp_.D}, "Emb"),
        logit_({hp_.B, hp_.Lm, hp_.V}, "Logit"),
        m_embedding_(name + ".embedding", hp_.V, hp_.D, token_, emb_),
        m_layers_(), m_output_(name + ".output", hp_.D, hp_.V, emb_, logit_) {
    this->logger_ = setup_logger("llama", "trace");

    add_submodule(m_embedding_);
    for (auto i = 0; i < hp_.Y; ++i) {
      auto l = layer(name + fmt::format(".layer{}", i), hp_, emb_);
      m_layers_.push_back(l);
      add_submodule(l);
    }
    add_submodule(m_output_);
    add_inout(token_);

    MDEBUG("New module of Llama:\n{}\n{}", this->hp_.str(), this->str());
  }
  void forward() override {
    m_embedding_.forward();
    for (auto layer : m_layers_) {
      layer.forward();
    }
    m_output_.forward();
    func::sampling_top_p(logit_, token_);
  }
}; // class Llama
   //
Llama<float> make_fp32_llama(const std::string name, Tensor<float> token);

} // namespace llama
