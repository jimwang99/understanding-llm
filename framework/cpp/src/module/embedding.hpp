#pragma once

#include "module.hpp"
#include "tensor.hpp"

namespace func {

template <typename T>
void embedding_out(const Tensor<T> &token, const Tensor<T> &weight,
                   Tensor<T> &output) {
  ASSERT(token.ndim() >= 1, token.str());
  ASSERT(weight.ndim() == 2, weight.str());
  auto dim = weight.shape(0);

  auto shape = token.shape();
  shape.insert(shape.begin(), dim);

  output.reshape(shape);

  for (size_t i = 0; i < token.size(); ++i) {
    for (size_t d = 0; d < dim; ++d) {
      output.at(i, d) = weight.at(d, token.at(i));
    }
  }
}  // FuncEmbedding

}  // namespace func

////////////////////////////////////////////////////////////////////////////////

namespace module {

template <typename T>
class Embedding : public virtual Module<T> {
 protected:
  size_t vocab_;
  size_t dim_;

  Tensor<T> embedding_;

  std::shared_ptr<Tensor<T>> input_;
  std::shared_ptr<Tensor<T>> output_;

 public:
  Embedding(std::string name, size_t vocab, size_t dim, Tensor<T> &input,
            Tensor<T> &output)
      : Module<T>(name),
        vocab_(vocab),
        dim_(dim),
        embedding_({vocab, dim}, "embedding"),
        input_(std::make_shared<Tensor<T>>(input)),
        output_(std::make_shared<Tensor<T>>(output)) {
    this->params_["Vocab"] = vocab_;
    this->params_["Dim"] = dim_;
    this->weights_.push_back(std::make_shared<Tensor<T>>(embedding_));
    this->inputs_.push_back(input_);
    this->outputs_.push_back(output_);

    this->logger_ = setup_logger("embedding");
    DEBUG("New module of Embedding:\n{}", this->str());
  }

  void forward() override {
    ASSERT(input_->shape(0) == vocab_, "vocab_={} input_={}", vocab_,
           input_.str());
    ASSERT(output_->shape(0) == dim_, "dim_={} output_={}", dim_,
           output_.str());
    func::embedding_out(*input_, embedding_, *output_);
  }
};  // class Embedding

}  // namespace module
