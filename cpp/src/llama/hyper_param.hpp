#pragma once

#include <fmt/format.h>

#include <cstddef>
#include <string>

#include "logger.hpp"

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
      : B(hp.B),
        D(hp.D),
        V(hp.V),
        Lm(hp.Lm),
        Y(hp.Y),
        Nh(hp.Nh),
        Nkv(hp.Nkv),
        M(hp.M),
        Dn(hp.Dn),
        Dh(hp.Dh),
        Dkv(hp.Dkv),
        Eps(hp.Eps) {}

  HyperParam(const size_t B, const size_t D, const size_t V, const size_t Lm,
             const size_t Y, const size_t Nh, const size_t Nkv, const size_t M,
             const double Eps = 1e-5)
      : B(B),
        D(D),
        V(V),
        Lm(Lm),
        Y(Y),
        Nh(Nh),
        Nkv(Nkv),
        M(M),
        Dn(CalcHiddenDim(D, M)),
        Dh(Dn / Nh),
        Dkv(Dn / Nkv),
        Eps(Eps) {}

  std::string str() const {
    return fmt::format(
        "HyperParam D={} V={} Lm={} Y={} Nh={} "
        "Nkv={} M={} Dn={} Dh={} Dkv={} Eps={}",
        D, V, Lm, Y, Nh, Nkv, M, Dn, Dh, Dkv, Eps);
  };
};

}  // namespace llama