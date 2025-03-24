#pragma once

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "logger.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using TensorShape = std::vector<size_t>;

enum class DType {
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
  kFP32,
  kFP64,
};

DType typeid_to_dtype(const std::string &typeid_str);

////////////////////////////////////////////////////////////////////////////////
// TensorBase
////////////////////////////////////////////////////////////////////////////////

class TensorBase {
protected:
  std::string name_;
  TensorShape shape_;
  TensorShape strides_;
  DType dtype_;

  static const std::map<DType, std::string> dtype_to_str_;
  static const std::map<DType, size_t> dtype_to_size_;

public:
  //============================================================================
  // constructors
  //============================================================================
  TensorBase(const std::string name = "", const DType dtype = DType::kFP32);
  virtual ~TensorBase() = default;

  //============================================================================
  // modifiers
  //============================================================================
  void set_name(const std::string name);
  void reshape(const TensorShape shape);
  void view(const TensorShape shape);

  //============================================================================
  // properties
  //============================================================================
  size_t size() const;
  size_t nbytes() const;
  size_t ndim() const;
  TensorShape shape() const;
  size_t shape(const size_t i) const;
  TensorShape stride() const;
  size_t stride(const size_t i) const;
  const std::string &name() const;
  DType dtype() const;
  const std::string &dtype_str() const;
  size_t elem_size() const;

  virtual void resize_data() = 0;
  virtual const void *buf() const = 0;
  virtual void *mutable_buf() = 0;
  virtual std::string str() const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// Tensor
////////////////////////////////////////////////////////////////////////////////

template <typename T> class Tensor : public TensorBase {
private:
  std::vector<T> vec_;

public:
  //============================================================================
  // constructors
  //============================================================================
  Tensor(const std::string name = "T", const TensorShape shape = {1},
         const std::vector<T> value = {0})
      : TensorBase(name, typeid_to_dtype(typeid(T).name())) {
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    ASSERT(sizeof(T) == elem_size(), "T must be the elem_size()");
    reshape(shape);
    size_t m = std::min(size(), value.size());
    for (size_t i = 0; i < m; ++i) {
      at(i) = value.at(i);
    }
  }

  // copy constructor
  Tensor(const Tensor<T> &other)
      : Tensor(other.name(), other.shape(), other.vec()) {}

  ~Tensor() = default;

  //----------------------------------------------------------------------------
  // initializers
  //----------------------------------------------------------------------------
  void zeros() { vec_.assign(vec_.size(), 0); }
  void ones() { vec_.assign(vec_.size(), 1); }
  void full(const T value) { vec_.assign(vec_.size(), value); }

  void linspace(const T start, const T step) {
    T v = start;
    for (size_t i = 0; i < size(); ++i) {
      at(i) = v;
      v += step;
    }
  }

  void rand(const T min, const T max) {
    if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
      for (size_t i = 0; i < size(); ++i) {
        at(i) = rand() / T(RAND_MAX) * (max - min) + min;
      }
    }
  }

  void operator=(const Tensor<T> &other) {
    set_name(other.name());
    reshape(other.shape());
    for (size_t i = 0; i < size(); ++i) {
      at(i) = other.at(i);
    }
  }

  //============================================================================
  // modifiers
  //============================================================================
  void resize_data() override { vec_.resize(size()); }

  //============================================================================
  // accessors
  //============================================================================
  T &at(const size_t i) { return vec_.at(i); }
  const T &at(const size_t i) const { return vec_.at(i); }
  T &at(const size_t i0, const size_t i1) {
    return vec_.at(i1 * strides_[1] + i0);
  }
  const T &at(const size_t i0, const size_t i1) const {
    return vec_.at(i1 * strides_[1] + i0);
  }
  T &at(const size_t i0, const size_t i1, const size_t i2) {
    return vec_.at(i2 * strides_[2] + i1 * strides_[1] + i0);
  }
  const T &at(const size_t i0, const size_t i1, const size_t i2) const {
    return vec_.at(i2 * strides_[2] + i1 * strides_[1] + i0);
  }
  T &at(const size_t i0, const size_t i1, const size_t i2, const size_t i3) {
    return vec_.at(i3 * strides_[3] + i2 * strides_[2] + i1 * strides_[1] + i0);
  }
  const T &at(const size_t i0, const size_t i1, const size_t i2,
              const size_t i3) const {
    return vec_.at(i3 * strides_[3] + i2 * strides_[2] + i1 * strides_[1] + i0);
  }

  const void *buf() const override { return vec_.data(); }
  void *mutable_buf() override { return vec_.data(); }

  const T *data() const { return vec_.data(); }
  T *mutable_data() { return vec_.data(); }

  const std::vector<T> &vec() const { return vec_; }
  std::vector<T> &mutable_vec() { return vec_; }

  //============================================================================
  // pretty print (for debugging purpose)
  //============================================================================
  std::string str() const override {
    std::string s =
        fmt::format("Tensor name={} shape=[{}]", name_, fmt::join(shape_, ","));
    s += " value=[";

    if (size() > 9) {
      auto sz = size();
      s += fmt::format("{}, {}, {}, {}, {}, {}, ..., {}, {}, {}]", at(0), at(1),
                       at(2), at(3), at(4), at(5), at(sz - 3), at(sz - 2),
                       at(sz - 1));
    } else {
      s += fmt::format("{}]", fmt::join(vec_, ", "));
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

template <typename T>
bool is_close(const Tensor<T> &lhs, const Tensor<T> &rhs, const T atol = 1e-16,
              const T rtol = 1e-6) {
  if (lhs.shape() != rhs.shape()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (std::abs(lhs.at(i) - rhs.at(i)) > atol + rtol * std::abs(rhs.at(i))) {
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

using TensorBasePtr = std::shared_ptr<TensorBase>;
using CTensorBasePtr = std::shared_ptr<const TensorBase>;

template <typename T> using TensorPtr = std::shared_ptr<Tensor<T>>;
template <typename T> using CTensorPtr = std::shared_ptr<const Tensor<T>>;