#include "tensor.hpp"
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// TensorBase
////////////////////////////////////////////////////////////////////////////////

// Initialize static members
const std::map<DType, std::string> TensorBase::dtype_to_str_ = {
    {DType::kFP32, "fp32"},     {DType::kFP64, "fp64"},
    {DType::kInt32, "int32"},   {DType::kInt64, "int64"},
    {DType::kUInt32, "uint32"}, {DType::kUInt64, "uint64"},
};

const std::map<DType, size_t> TensorBase::dtype_to_size_ = {
    {DType::kFP32, 4},  {DType::kFP64, 8},   {DType::kInt32, 4},
    {DType::kInt64, 8}, {DType::kUInt32, 4}, {DType::kUInt64, 8},
};

DType typeid_to_dtype(const std::string &typeid_str) {
  static const std::map<std::string, DType> typeid_to_dtype_ = {
      {"i", DType::kInt32},  {"l", DType::kInt64}, {"j", DType::kUInt32},
      {"m", DType::kUInt64}, {"f", DType::kFP32},  {"d", DType::kFP64},
  };
  return typeid_to_dtype_.at(typeid_str);
}

//============================================================================
// constructors
//============================================================================
TensorBase::TensorBase(const std::string name, const DType dtype)
    : name_(name), shape_(), strides_(), dtype_(dtype) {}

//============================================================================
// modifiers
//============================================================================
void TensorBase::set_name(const std::string name) { name_ = name; }

void TensorBase::reshape(const TensorShape shape) {
  if (shape_ == shape) {
    return;
  }
  shape_ = shape;
  strides_.resize(shape_.size() + 1);
  strides_[0] = 1;
  for (size_t i = 1; i <= shape_.size(); ++i) {
    strides_[i] = strides_[i - 1] * shape_[i - 1];
  }
  this->resize_data();
}

void TensorBase::view(const TensorShape shape) {
  size_t new_size = 1;
  for (auto s : shape) {
    new_size *= s;
  }
  assert(new_size == size());
  reshape(shape);
}

//============================================================================
// properties
//============================================================================
size_t TensorBase::size() const { return strides_.back(); }
size_t TensorBase::nbytes() const { return size() * dtype_to_size_.at(dtype_); }
size_t TensorBase::ndim() const { return shape_.size(); }
TensorShape TensorBase::shape() const { return shape_; }
TensorShape TensorBase::stride() const { return strides_; }
size_t TensorBase::shape(const size_t i) const {
  assert(i < shape_.size());
  return shape_[i];
}
size_t TensorBase::stride(const size_t i) const {
  assert(i <= strides_.size());
  return strides_[i];
}
const std::string &TensorBase::name() const { return name_; }
DType TensorBase::dtype() const { return dtype_; }
const std::string &TensorBase::dtype_str() const {
  return dtype_to_str_.at(dtype_);
}
size_t TensorBase::elem_size() const { return dtype_to_size_.at(dtype_); }
