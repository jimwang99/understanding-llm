#include "func/elem_wise.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

#include "logger.hpp"

template <typename T>
void test_inline(std::function<void(Tensor<T> &, Tensor<T> &)> f,
                 const Tensor<T> &z) {
  Tensor<T> x({3}, "x");
  x.linspace(1.0, 0.5);
  Tensor<T> y({3}, "y");
  y.linspace(2.0, 0.5);
  f(x, y);
  EXPECT_EQ(x, z);
}

TEST(test_elem_wise, float) {
  {
    Tensor<float> z({3}, {3.0, 4.0, 5.0}, "z");
    test_inline<float>(func::add_inline<float>, z);
  }
  {
    Tensor<float> z({3}, {-1.0, -1.0, -1.0}, "z");
    test_inline<float>(func::sub_inline<float>, z);
  }
  {
    Tensor<float> z({3}, {2.0, 3.75, 6.0}, "z");
    test_inline<float>(func::mul_inline<float>, z);
  }
  {
    Tensor<float> z({3}, {1.0 / 2.0, 1.5 / 2.5, 2.0 / 3.0}, "z");
    test_inline<float>(func::div_inline<float>, z);
  }
}

int main(int argc, char **argv) {
  setup_logger("elem_wise.hpp", "trace");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
