#include "llama.hpp"

#include "llama/hyper_param.hpp"

namespace llama {

constexpr HyperParam get_hyper_param(const std::string name) {
  if (name == "tinystories260k") {
    return constexpr HyperParam(1, 64, 512, 512, 5, 8, 4, 8, 1e-5);
  }
  throw std::runtime_error("Unsupported model name " + name);
}

Llama<float> make_fp32_llama(const std::string name, Tensor<float>& token) {
  auto hp = get_hyper_param(name);
  token.reshape({hp.B, hp.Lm});
  return Llama<float>(name, hp, token);
}

}  // namespace llama