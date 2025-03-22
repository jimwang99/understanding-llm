#pragma once

#include "tensor.hpp"

namespace func {

template <typename T>
void sampling_greedy(const Tensor<T> &logit, Tensor<T> &token) {
}

template <typename T, int K>
void sampling_top_k(const Tensor<T> &logit, Tensor<T> &token) {}

template <typename T, int K>
void sampling_top_p(const Tensor<T> &logit, Tensor<T> &token) {}

}  // namespace func