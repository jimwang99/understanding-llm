#pragma once

#include <cassert>
#include <cstdio>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "spdlog/spdlog.h"

////////////////////////////////////////////////////////////////////////////////
// Assertion with message
#define assertm(condition, ...)                                                \
  assert(condition || (spdlog::error(__VA_ARGS__) &&                           \
                       spdlog::error(" at %s:%d\n", __FILE__, __LINE__)))

////////////////////////////////////////////////////////////////////////////////
// Get type name as string
namespace {
template <typename T> struct TypeParseTraits;

#define REGISTER_PARSE_TYPE(X)                                                 \
  template <> struct TypeParseTraits<X> {                                      \
    static const char *name;                                                   \
  };                                                                           \
  const char *TypeParseTraits<X>::name = #X

REGISTER_PARSE_TYPE(float);
REGISTER_PARSE_TYPE(double);
REGISTER_PARSE_TYPE(uint32_t);
REGISTER_PARSE_TYPE(uint16_t);
REGISTER_PARSE_TYPE(uint8_t);
REGISTER_PARSE_TYPE(int32_t);
REGISTER_PARSE_TYPE(int16_t);
REGISTER_PARSE_TYPE(int8_t);
} // unnamed namespace

template <typename T> const char *getTypeName() {
  return TypeParseTraits<T>::name;
}

template <typename T> bool isType(const char *type_name) {
  return (0 == strcmp(TypeParseTraits<T>::name, type_name));
}

template <typename Ta, typename Tb> bool isType() {
  return (0 == strcmp(TypeParseTraits<Ta>::name, TypeParseTraits<Tb>::name));
}

////////////////////////////////////////////////////////////////////////////////
// Tensor class
template <typename T> class Tensor4 {
public:
  // Constructor
  Tensor4(std::vector<size_t> shape)
      : _shape(4, 1), _stride_1(0), _shape_012(0), _shape_all(0), _data(1) {
    view(shape, is_init = true);
    _data.resize(numel());
  }

  // Constructor with initialization value
  Tensor4(std::vector<size_t> shape, const T &value) : Tensor(shape) {
    init(value);
  }

  // View as shape
  void view(std::vector<size_t> shape, bool is_init = false) {
    assertm(shape.size() <= 4, "%lu != 4", shape.size());
    while
      shape.size() < 4 { shape.push_front(1); }
    spdlog::trace("Tensor4::view shape=({}, {}, {}, {})", shape[3], shape[2],
                  shape[1], shape[0]);

    if (!is_init) {
      shape_all = shape[0] * shape[1] * shape[2] * shape[3];
      assertm(shape_all == _shape_all, "%lu != %lu", shape_all, _shape_all);
    }

    _shape = shape;

    _stride_1 = _shape[0] * _shape[1];
    _shape_012 = _stride_1 * _shape[2];
    _shape_all = _shape_012 * _shape[3];
  }

  // Initialize with value
  void init(const T &value) { std::fill(_data.begin(), _data.end(), value); }

  // Total number of elements
  inline size_t numel() { return _sz_all; }

  // Shape
  inline std::vector<size_t> shape() { return _shape; }

  // number of byptes
  inline size_t nbytes() { return numel() * sizeof(T); }

  // strides
  inline size_t stride_1() { return _shape[0]; }
  inline size_t stride_2() { return _stride_1; }
  inline size_t stride_3() { return _shape_012; }

  //============================================================================
  // Accessor
  T *data() { return _data.data(); }
  const T *data() const { return _data.data(); }

  T &at(const std::vector<size_t> idx) {
    assertm(idx.size() <= 4, "%lu != 4", idx.size());
    while
      idx.size() < 4 { idx.push_front(0); }
    assertm(idx[0] < _shape[0], "idx[0]=%lu >= _shape[0]=%lu", idx[0],
            _shape[0]);
    assertm(idx[1] < _shape[1], "idx[1]=%lu >= _shape[1]=%lu", idx[1],
            _shape[1]);
    assertm(idx[2] < _shape[2], "idx[2]=%lu >= _shape[2]=%lu", idx[2],
            _shape[2]);
    assertm(idx[3] < _shape[3], "idx[3]=%lu >= _shape[3]=%lu", idx[3],
            _shape[3]);
    auto i =
        idx[3] * _shape_012 + idx[2] * _stride_1 + idx[1] * _shape[0] + idx[0];
    return _data[idx];
  }

  T &at(const size_t idx) {
    assertm(idx < _shape_all, "idx=%lu >= _shape_all=%lu", idx, _shape_all);
    return _data[idx];
  }

  std::string repr(bool brief = true) {
    std::ostringstream buf;
    buf << "Tensor (" << _sz0 << ", " << _sz1 << ", " << _sz2 << ", " << _sz3
        << ")" << std::endl;
    uint32_t i0 = 0;
    while (i0 < _sz0) {
      // PRINT_DEBUG("i0=%d", i0);
      if (brief && (_sz0 > 2) && (i0 == 1)) {
        i0 = _sz0 - 1;
        buf << "............ " << std::endl;
        continue;
      }
      auto idx0 = i0 * _sz_123;

      uint32_t i1 = 0;
      while (i1 < _sz1) {
        if (brief && (_sz1 > 2) && (i1 == 1)) {
          i1 = _sz1 - 1;
          buf << "......... " << std::endl;
          continue;
        }

        auto idx01 = idx0 + i1 * _sz_23;

        uint32_t i2 = 0;
        while (i2 < _sz2) {
          if (brief && (_sz2 > 3) && (i2 == 1)) {
            i2 = _sz2 - 1;
            buf << "...... " << std::endl;
            continue;
          }

          auto idx012 = idx01 + i2 * _sz3;
          buf << "[" << std::setw(3) << i0 << ", " << std::setw(3) << i1 << ", "
              << std::setw(3) << i2 << ", _] ";

          uint32_t i3 = 0;
          while (i3 < _sz3) {
            if (brief && (_sz3 > 5) && (i3 == 4)) {
              i3 = _sz3 - 1;
              buf << "... ";
              continue;
            }
            auto idx0123 = idx012 + i3;
            buf << _ptr[idx0123] << " ";
            ++i3;
          }
          buf << std::endl;
          ++i2;
        }
        ++i1;
      }
      ++i0;
    }
    return buf.str();
  }

private:
  // size of each dimension
  std::vector<size_t> _shape;  // 0: innermost, 3: outermost
  std::vector<size_t> _stride; // 0: innermost, 3: outermost

  // storage
  std::vector<T> _data;
};
