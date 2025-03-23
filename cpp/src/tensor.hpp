#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "logger.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using TensorShape = std::vector<size_t>;

template <typename T> class Tensor {
private:
  std::string name_;
  std::vector<T> data_;
  TensorShape shape_;
  TensorShape strides_;

public:
  //============================================================================
  // constructors
  //============================================================================
  // default constructor
  Tensor() : name_(), data_(), shape_(), strides_() { reshape({1}); }

  Tensor(const std::string name) : name_(name), data_(), shape_(), strides_() {
    reshape({1});
  }
  // constructor with shape
  Tensor(const std::string name, const TensorShape shape)
      : name_(name), data_(), shape_(), strides_() {
    reshape(shape);
  }
  // constructor with shape and iniital values
  Tensor(const std::string name, const TensorShape shape,
         const std::vector<T> value)
      : name_(name), data_(), shape_(), strides_() {
    reshape(shape);
    for (size_t i = 0; i < value.size(); ++i) {
      at(i) = value.at(i);
    }
  }

  //============================================================================
  // modifiers
  //============================================================================
  // reshape: change the shape of the tensor with possible memory reallocation
  void reshape(const TensorShape shape) {
    if (shape_ == shape) {
      return;
    }
    shape_ = shape;
    strides_.resize(shape_.size() + 1);
    strides_[0] = 1;
    for (int i = 1; i <= shape_.size(); ++i) {
      strides_[i] = strides_[i - 1] * shape_[i - 1];
    }
    data_.resize(strides_.back());
  }
  // view: change the shape of the tensor without changing the data
  void view(const TensorShape shape) {
    size_t new_size = 1;
    for (auto s : shape) {
      new_size *= s;
    }
    assert(new_size == size());
    reshape(shape);
  }

  void set_name(const std::string name) { name_ = name; }

  //----------------------------------------------------------------------------
  // initializers
  //----------------------------------------------------------------------------
  // fill the tensor with zeros
  void zeros() { data_.assign(data_.size(), 0); }
  // fill the tensor with linearly spaced values
  void linspace(const T start, const T step) {
    T v = start;
    for (size_t i = 0; i < size(); ++i) {
      at(i) = v;
      v += step;
    }
  }
  // fill the tensor with random values
  void rand(const T min, const T max) {
    if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
      for (size_t i = 0; i < size(); ++i) {
        at(i) = rand() / T(RAND_MAX) * (max - min) + min;
      }
    }
    throw std::runtime_error("Not implemented");
  }

  void operator=(const Tensor<T> &other) {
    set_name(other.name());
    reshape(other.shape());
    for (size_t i = 0; i < size(); ++i) {
      at(i) = other.at(i);
    }
  }

  //============================================================================
  // accessors
  //============================================================================
  T &at(const size_t i) { return data_.at(i); }
  const T &at(const size_t i) const { return data_.at(i); }
  T &at(const size_t i0, const size_t i1) {
    return data_.at(i1 * strides_[1] + i0);
  }
  const T &at(const size_t i0, const size_t i1) const {
    return data_.at(i1 * strides_[1] + i0);
  }
  T &at(const size_t i0, const size_t i1, const size_t i2) {
    return data_.at(i2 * strides_[2] + i1 * strides_[1] + i0);
  }
  const T &at(const size_t i0, const size_t i1, const size_t i2) const {
    return data_.at(i2 * strides_[2] + i1 * strides_[1] + i0);
  }
  T &at(const size_t i0, const size_t i1, const size_t i2, const size_t i3) {
    return data_.at(i3 * strides_[3] + i2 * strides_[2] + i1 * strides_[1] +
                    i0);
  }
  const T &at(const size_t i0, const size_t i1, const size_t i2,
              const size_t i3) const {
    return data_.at(i3 * strides_[3] + i2 * strides_[2] + i1 * strides_[1] +
                    i0);
  }

  const T *data() const { return data_.data(); }
  T *mutable_data() { return data_.data(); }

  //============================================================================
  // properties
  //============================================================================
  size_t size() const { return strides_.back(); }
  size_t nbytes() const { return data_.size() * sizeof(T); }
  TensorShape shape() const { return shape_; }
  TensorShape stride() const { return strides_; }
  size_t ndim() const { return shape_.size(); }
  size_t shape(const int i) const {
    assert(i < shape_.size());
    return shape_[i];
  }
  size_t stride(const int i) const {
    assert(i <= strides_.size());
    return strides_[i];
  }
  const std::string &name() const { return name_; }

  //============================================================================
  // pretty print (for debugging purpose)
  //============================================================================
  const std::string str() const {
    std::string s =
        fmt::format("Tensor name={} shape=[{}]", name_, fmt::join(shape_, ","));
    s += " value=[";

    if (size() > 6) {
      auto sz = size();
      s += fmt::format("{}, {}, {}, ..., {}, {}, {}]", at(0), at(1), at(2),
                       at(sz - 3), at(sz - 2), at(sz - 1));
    } else {
      for (size_t i = 0; i < size(); ++i) {
        s += fmt::format("{}, ", at(i));
      }
      s.pop_back();
      s.back() = ']';
    }
    return s;
  }
};

// comparison operator
template <typename T>
bool operator==(const Tensor<T> &lhs, const Tensor<T> &rhs) {
  if (lhs.shape() != rhs.shape()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs.at(i) != rhs.at(i)) {
      return false;
    }
  }
  return true;
}

// ostream operator (for gtest)
template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &obj) {
  os << obj.str();
  return os;
}

template <typename T> using TensorPtr = std::shared_ptr<Tensor<T>>;
template <typename T> using CTensorPtr = std::shared_ptr<const Tensor<T>>;