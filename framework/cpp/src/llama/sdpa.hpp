#pragma once

#include <cassert>
#include <cmath>
#include "tensor.hpp"
#include "matmul.hpp"
#include "add.hpp"
#include "div.hpp"

template <typename T>
Tensor<T> func_sdpa(const Tensor<T> &q, const Tensor<T> &k, const Tensor<T> &v, const Tensor<T> &mask) {
    assert(q.shape().size() == 4);
    assert(q.shape() == k.shape());
    assert(q.shape() == v.shape());
    
    auto B = q.shape(3);
    auto N = q.shape(2);
    auto L = q.shape(1);
    auto D = q.shape(0);

    assert(mask.shape().size() == 4);
    assert(mask.shape(0) == L);
    assert(mask.shape(1) == L);
    assert(mask.shape(2) == 1);
    assert(mask.shape(3) == 1);
    
    // Q⋅Kᵀ
    auto qk = func_matmul<T, true>(q, k);
    
    // Q⋅Kᵀ/√D
    func_div_scalar<T, true>(qk, std::sqrt(D));
    
    // mask(Q⋅Kᵀ/√D)
    auto m = func_broadcast_add<T, 2>(qk, mask);
    
    // softmax(mask(Q⋅Kᵀ/√D))
    auto sm = func_softmax(m);
    
    // V⋅softmax(mask(Q⋅Kᵀ/√D))
    auto a = func_matmul<T, false>(sm, v);
    
    assert (a.shape().size() == 4);
    assert (a.shape(3) == B);
    assert (a.shape(2) == N);
    assert (a.shape(1) == L);
    assert (a.shape(0) == D);
    
    return a;
}