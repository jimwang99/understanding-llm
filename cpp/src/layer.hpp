#pragma once

#include "input_norm.hpp"
#include "input_proj.hpp"
#include "module.hpp"

namespace llama {
template <typename T> class Layer final : public virtual Module<T> {
private:
  const HyperParam &hp_;
  const size_t y_; /*! layer index */

  Tensor<T> &emb_; /*! input and output of Layer*/

  Tensor<T> &cos_;  /*! cos constant for RoPE */
  Tensor<T> &sin_;  /*! sin constant for RoPE */
  Tensor<T> &mask_; /*! mask constant for SDPA */

  // hidden states of Layer
  Tensor<T> embn_;
  Tensor<T> q_;
  Tensor<T> k_;
  Tensor<T> v_;
  Tensor<T> cache_k_;
  Tensor<T> cache_v_;
  Tensor<T> attn_;
  Tensor<T> up_;
  Tensor<T> gate_;

  Linear<T> m_input_proj_q_; /*! linear module for query */
  Linear<T> m_input_proj_k_; /*! linear module for key */
  Linear<T> m_input_proj_v_; /*! linear module for value */

  KVCache<T> m_k_cache_; /*! key-value cache module for K */
  KVCache<T> m_v_cache_; /*! key-value cache module for V */

  LinearAdd<T> m_attn_proj_;  /*! module of attention projection and
                                residual add fusion */
  Linear<T> m_up_proj_;       /*! module of up projection */
  LinearSilu<T> m_gate_proj_; /*! module of gate projection and silu fusion */
  LinearAdd<T>
      m_down_proj_; /*! module of down projection and residual add fusion */

public:
  Layer(const std::string &name, const HyperParam &hp, const size_t y,
        Tensor<T> &emb, const Tensor<T> &cos, const Tensor<T> &sin,
        const Tensor<T> &mask)
      : Module<T>(name), hp_(hp), y_(y), emb_(emb), cos_(cos), sin_(sin),
        mask_(mask), embn_({hp_.B, hp_.Lm, hp_.D}, "emb_norm"),
        q_({hp_.B, hp_.Lm, hp_.Nh, hp_.Dh}, "q"),
        k_({hp_.B, hp_.Lm, hp_.Nkv, hp_.Dh}, "k"),
        v_({hp_.B, hp_.Lm, hp_.Nkv, hp_.Dh}, "v"),
        cache_k_({hp_.B, hp_.Lm, hp_.Nkv * hp_.Dh}, "cache_k"),
        cache_v_({hp_.B, hp_.Lm, hp_.Nkv * hp_.Dh}, "cache_v"),
        attn_({hp_.B, hp_.Nh, hp_.Lm, hp_.Lm}, "attention"),
        up_({hp_.B, hp_.Lm, hp_.Dn}, "up_proj"),
        gate_({hp_.B, hp_.Lm, hp_.Dn}, "gate_proj"),

        m_input_proj_q_(name + ".input_proj_q", hp_.D, hp_.D, false, emb, q_),
        m_input_proj_k_(name + ".input_proj_k", hp_.D, hp_.D * hp_.Nkv / hp_.Nh,
                        false, emb, k_),
        m_input_proj_v_(name + ".input_proj_v", hp_.D, hp_.D * hp_.Nkv / hp_.Nh,
                        false, emb, v_),
        m_k_cache_(name + ".k_cache", hp_.Lm, hp_.Nkv * hp_.Dh, cache_k_, k_),
        m_v_cache_(name + ".v_cache", hp_.Lm, hp_.Nkv * hp_.Dh, cache_v_, v_),
        m_attn_proj_(name + ".attn_proj", hp_.D, hp_.D, q_, emb_, emb_),
        m_up_proj_(name + ".up_proj", hp_.D, hp_.Dn, embn_, up_),
        m_gate_proj_(name + ".gate_proj", hp_.D, hp_.Dn, embn_, gate_),
        m_down_proj_(name + ".down_proj", hp_.D, hp_.Dn, up_, emb_),
        logger_(get_logger("Layer")),

  {
    add_inout(emb_);
    add_submodule(m_input_proj_q_);
    add_submodule(m_input_proj_k_);
    add_submodule(m_input_proj_v_);
    add_submodule(m_k_cache_);
    add_submodule(m_v_cache_);
    INFO("New module of Layer:\n{}", this->str());
  }

  void forward() override {
    // Input normalization
    func::rms_norm_out(emb_, embn_, hp_.Eps);

    // input projection
    m_input_proj_q_.forward();
    m_input_proj_k_.forward();
    m_input_proj_v_.forward();

    // rope
    func::rope_inline(q_, cos_, sin_); // TODO: decode stage
    func::rope_inline(k_, cos_, sin_); // TODO: decode stage

    // kv-cache
    m_k_cache_.forward();
    m_v_cache_.forward();

    // qkv-transpose
    func::transpose_1_2_inline<T>(q_);
    func::transpose_1_2_inline<T>(k_);
    func::transpose_1_2_inline<T>(v_);

    // SDPA: scaled-dot-product-attention
    func::spda_out<T>(q_, k_, v_, attn_, mask_);

    // attention projection
    func::transpose_1_2_inline<T>(q_);
    m_attn_proj_.forward();

    // FFN: feed-forward network
    func::rms_norm_out(emb_, embn_, hp_.Eps);
    m_up_proj_.forward();
    m_gate_proj_.forward();
    func::mul_inline<T>(up_, gate_);
    m_down_proj_.forward();
  }
};
} // namespace llama
