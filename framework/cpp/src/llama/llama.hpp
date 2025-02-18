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
#include "logger.hpp"
#include "module.hpp"
#include "module/embedding.hpp"

namespace llama {

using namespace module;

template <typename T>
class Llama : public virtual Module<T> {
 private:
  HyperParam hp_;

  Tensor<T> token_;
  Tensor<T> emb_;
  Tensor<T> logit_;

  Embedding<T> m_embedding_;
  std::vector<Layer<T>> m_layers_;

  std::shared_ptr<spdlog::logger> logger_;

 public:
  Llama(const std::string name, const HyperParam &hp)
      : Module<T>(name),
        hp_(hp),
        token_({hp_.B, hp.Lm}, "Token"),
        emb_({hp_.B, hp_.Lm, hp_.D}, "Emb"),
        logit_({hp_.B, hp_.Lm, hp_.V}, "Logit"),
        m_embedding_(name + ".embedding", hp_.V, hp_.D, token_, emb_),
        m_layers_() {
    logger_ = setup_logger("llama");
    DEBUG("New module of Llama:\n{}\n{}", this->hp_.str(), this->str());
  }
  void forward() override {}
};  // class Llama
    //
Llama<float> make_fp32_llama(const std::string name);

}  // namespace llama
