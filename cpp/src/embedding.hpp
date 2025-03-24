#pragma once

#include "module.hpp"
#include "tensor.hpp"

namespace func {
template <typename T>
void embedding_out(const Tensor<T> &token, const Tensor<T> &weight,
                   Tensor<T> &output) {
  auto dim = weight.shape(0);
  auto shape = token.shape();
  shape.insert(shape.begin(), dim);

  output.reshape(shape);

  for (size_t i = 0; i < token.size(); ++i) {
    for (size_t d = 0; d < dim; ++d) {
      output.at(i, d) = weight.at(d, token.at(i));
    }
  }
} // embedding_out
} // namespace func

////////////////////////////////////////////////////////////////////////////////

namespace module {

template <typename T> class Embedding : public virtual Module<T> {
protected:
  size_t vocab_;
  size_t dim_;

  CTensorPtr<size_t> token_;
  TensorPtr<T> emb_;
  TensorPtr<T> weight_;

public:
  Embedding(const std::string name, const size_t vocab, const size_t dim,
            CTensorPtr<size_t> token, TensorPtr<T> emb)
      : Module<T>(name), vocab_(vocab), dim_(dim), token_(token), emb_(emb),
        weight_("weight", {vocab, dim}), logger_(get_logger("embedding")) {
    add_param("vocab", vocab_);
    add_param("dim", dim_);
    add_weight(weight_);
    add_input(token_);
    add_output(output_);
    DEBUG("New module of Embedding:\n{}", this->str());
  }

  void forward() override {
    ASSERT(input_.shape(0) == vocab_, "vocab_={} input_={}", vocab_,
           input_.str());
    ASSERT(output_.shape(0) == dim_, "dim_={} output_={}", dim_, output_.str());
    func::embedding_out(input_, embedding_, output_);
  }
}; // class Embedding

} // namespace module
