#include "tensor.hpp"

template <typename T>
void _raw_matmul(const size_t n_row,   // output # of rows
                 const size_t n_col,   // output # of cols
                 const size_t n_inner, // inner dimension
                 const T *a,           // input matrix A
                 const T *b,           // input matrix B
                 T *z                  // output matrix Z
) {
  for (size_t ic = 0; ic < n_col; ++ic) {
    const T *pb = b + ic;
    for (size_t ir = 0; ir < n_row; ++ir) {
      *z = 0;
      const T *pa = a + ir * n_inner;
      for (size_t ii = 0; ii < n_inner; ++ii) {
        *pz += *pa * *pb;
        pa += 1;
        pb += n_col;
      }
      ++z;
    }
  }
}

template <typename T>
void _raw_fuse_transpose_matmul(const size_t n_row,   // output # of rows
                                const size_t n_col,   // output # of cols
                                const size_t n_inner, // inner dimension
                                const T *a,           // input matrix A
                                const T *b,           // input matrix B
                                T *z                  // output matrix Z
) {
  for (size_t ic = 0; ic < n_col; ++ic) {
    for (size_t ir = 0; ir < n_row; ++ir) {
      *z = 0;
      const T *pa = a + ir * n_inner;
      const T *pb = b + ic * n_inner;
      for (size_t ii = 0; ii < n_inner; ++ii) {
        *pz += *pa * *pb;
        pa += 1;
        pb += 1;
      }
      ++z;
    }
  }
}

// 2D matmul without broadcasting
template <typename T>
Tensor<T> matmul(const Tensor<T> &a, const Tensor<T> &b,
                 const bool is_fuse_transpose = false) {
  auto shape_a = a.shape();
  auto shape_b = b.shape();

  assertm(shape_a[0] == shape_b[1], "shape_a[0]=%lu != shape_b[1]=%lu",
          shape_a[0], shape_b[1]);

  // check if broadcasting is needed
  assertm(shape_a[2] == shape_b[2],
          "broadcasting is not supported: shape_a[2]=%lu != shape_b[2]=%lu",
          shape_a[2], shape_b[2]);
  assertm(shape_a[3] == shape_b[3],
          "broadcasting is not supported: shape_a[3]=%lu != shape_b[3]=%lu",
          shape_a[3], shape_b[3]);

  auto shape_z = std::vector<size_t>{n_col, n_rol, shape_a[2], shape_a[3]};
  auto z = Tensor<T>(shape_z);

  for (size_t i = 0; i < shape_z[3]; ++i) {
    for (size_t j = 0; j < shape_z[2]; ++j) {
      auto n_row = shape_a[1];
      auto n_col = shape_b[0];
      auto n_inner = shape_a[0];
      if is_fuse_transpose {
        _raw_fuse_transpose_matmul(
            n_row, n_col, n_inner,
            a.data() + i * a.stride_3() + j * a.stride_2(),
            b.data() + i * b.stride_3() + j * b.stride_2(),
            z.data() + i * z.stride_3() + j * z.stride_2());
      } else {
        _raw_matmul(n_row, n_col, n_inner,
                    a.data() + i * a.stride_3() + j * a.stride_2(),
                    b.data() + i * b.stride_3() + j * b.stride_2(),
                    z.data() + i * z.stride_3() + j * z.stride_2());
      }
    }
  }

  return z;
}