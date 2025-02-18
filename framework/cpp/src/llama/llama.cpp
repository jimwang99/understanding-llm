#include "llama.hpp"

#include "llama/hyper_param.hpp"

namespace llama {

HyperParam get_hyper_param(const std::string name) {
  if (name == "tinystories260k") {
    return HyperParam(1, 64, 512, 512, 5, 8, 4, 8, 1e-5);
  }
  throw std::runtime_error("Unsupported model name " + name);
}

Llama<float> make_fp32_llama(const std::string name) {
  return Llama<float>(name, get_hyper_param(name));
}

}  // namespace llama