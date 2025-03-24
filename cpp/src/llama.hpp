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

#include "embedding.hpp"
#include "layer.hpp"
#include "linear.hpp"
#include "logger.hpp"
#include "module.hpp"

#include <fmt/format.h>

#include <cstddef>
#include <string>

namespace llama {

static constexpr size_t CalcHiddenDim(size_t D, size_t M) {
  return ((8 * D / 3 + M - 1) / M) * M;
}

struct HyperParam {
  size_t B;   /*! batch size */
  size_t D;   /*! embedding dimension */
  size_t V;   /*! vocab size */
  size_t Lm;  /*! maximum sequence length */
  size_t Y;   /*! number of layers */
  size_t Nh;  /*! number of attention heads */
  size_t Nkv; /*! number of key/value heads */
  size_t M;   /*! rounding to multiple of M*/
  size_t Dn;  /*! hidden dimension */
  size_t Dh;  /*! dimension per head */
  size_t Dkv; /*! dimension per key/value head */
  double Eps; /*! epsilon for normalization */

  // copy constructor
  HyperParam(const HyperParam &hp)
      : B(hp.B), D(hp.D), V(hp.V), Lm(hp.Lm), Y(hp.Y), Nh(hp.Nh), Nkv(hp.Nkv),
        M(hp.M), Dn(hp.Dn), Dh(hp.Dh), Dkv(hp.Dkv), Eps(hp.Eps) {}

  HyperParam(const size_t B, const size_t D, const size_t V, const size_t Lm,
             const size_t Y, const size_t Nh, const size_t Nkv, const size_t M,
             const double Eps = 1e-5)
      : B(B), D(D), V(V), Lm(Lm), Y(Y), Nh(Nh), Nkv(Nkv), M(M),
        Dn(CalcHiddenDim(D, M)), Dh(Dn / Nh), Dkv(Dn / Nkv), Eps(Eps) {}

  std::string str() const {
    return fmt::format("HyperParam D={} V={} Lm={} Y={} Nh={} "
                       "Nkv={} M={} Dn={} Dh={} Dkv={} Eps={}",
                       D, V, Lm, Y, Nh, Nkv, M, Dn, Dh, Dkv, Eps);
  };
};

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
    this->logger_ = get_logger("llama", "trace");

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
