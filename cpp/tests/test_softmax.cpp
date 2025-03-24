#include "../src/softmax.hpp"
#include "../src/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

class SoftmaxTest : public ::testing::Test {
protected:
  void SetUp() override { get_logger(); }
};

TEST_F(SoftmaxTest, BasicSoftmax) {
  Tensor<float> x("input", {3, 2}, {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f});

  TRACE("before: {}", x.str());
  func::softmax_inline(x);
  TRACE("after: {}", x.str());

  for (size_t b = 0; b < 2; ++b) {
    float sum = 0;
    for (size_t i = 0; i < 3; ++i) {
      sum += x.at(i, b);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6);
    EXPECT_TRUE(x.at(0, b) < x.at(1, b));
    EXPECT_TRUE(x.at(1, b) < x.at(2, b));
  }
}

// Test softmax with extreme values
TEST_F(SoftmaxTest, ExtremeValues) {
  Tensor<float> x("extreme", {3}, {1000.0f, 0.0f, -1000.0f});

  func::softmax_inline(x);

  EXPECT_NEAR(x.at(0), 1.0f, 1e-6);
  EXPECT_NEAR(x.at(1), 0.0f, 1e-6);
  EXPECT_NEAR(x.at(2), 0.0f, 1e-6);

  float sum = x.at(0, 0) + x.at(1, 0) + x.at(2, 0);
  EXPECT_NEAR(sum, 1.0f, 1e-6);
}

TEST_F(SoftmaxTest, UniformValues) {
  Tensor<float> x("uniform", {4}, {1.0f, 1.0f, 1.0f, 1.0f});

  func::softmax_inline(x);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(x.at(i), 0.25f, 1e-6);
  }
}

TEST_F(SoftmaxTest, ZeroValues) {
  Tensor<float> x("zeros", {3}, {0.0f, 0.0f, 0.0f});

  func::softmax_inline(x);

  float expected = 1.0f / 3.0f;
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(x.at(i), expected, 1e-6);
  }
}

TEST_F(SoftmaxTest, SafeVsUnsafe) {
  Tensor<float> safe("safe", {3}, {0.1f, 0.2f, 0.3f});
  Tensor<float> unsafe("unsafe", {3}, {0.1f, 0.2f, 0.3f});

  func::softmax_inline(safe, true);    // safe mode
  func::softmax_inline(unsafe, false); // unsafe mode

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(safe.at(i), unsafe.at(i), 1e-6);
  }

  float sum_safe = safe.at(0) + safe.at(1) + safe.at(2);
  float sum_unsafe = unsafe.at(0) + unsafe.at(1) + unsafe.at(2);
  EXPECT_NEAR(sum_safe, 1.0f, 1e-6);
  EXPECT_NEAR(sum_unsafe, 1.0f, 1e-6);
}

TEST_F(SoftmaxTest, SafeVsUnsafe2) {
  Tensor<float> safe("safe", {3}, {100.0f, 101.0f, 102.0f});
  Tensor<float> unsafe("unsafe", {3}, {100.0f, 101.0f, 102.0f});

  func::softmax_inline(safe, true);    // safe mode
  func::softmax_inline(unsafe, false); // unsafe mode

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_NE(safe.at(i), unsafe.at(i));
    EXPECT_TRUE(std::isnan(unsafe.at(i)));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
