#include "../src/activation.hpp"
#include "../src/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Test direct sigmoid function
TEST(ActivationTest, SigmoidFunction) {
  // Test sigmoid(0) = 0.5
  EXPECT_NEAR(func::sigmoid(0.0f), 0.5f, 1e-6);

  // Test sigmoid for large positive number approaches 1
  EXPECT_NEAR(func::sigmoid(10.0f), 1.0f, 1e-4);

  // Test sigmoid for large negative number approaches 0
  EXPECT_NEAR(func::sigmoid(-10.0f), 0.0f, 1e-4);

  // Test sigmoid symmetry around 0
  float x = 2.0f;
  EXPECT_NEAR(func::sigmoid(x) + func::sigmoid(-x), 1.0f, 1e-6);
}

// Test direct silu function
TEST(ActivationTest, SiluFunction) {
  // Test silu(0) = 0
  EXPECT_NEAR(func::silu(0.0f), 0.0f, 1e-6);

  // Test silu for positive number
  float x = 2.0f;
  EXPECT_NEAR(func::silu(x), x * func::sigmoid(x), 1e-6);

  // Test silu for negative number
  x = -2.0f;
  EXPECT_NEAR(func::silu(x), x * func::sigmoid(x), 1e-6);
}

// Test sigmoid tensor operations
TEST(ActivationTest, SigmoidTensor) {
  // Create input tensor
  Tensor<float> input("input", {2, 2});
  input.at(0) = 0.0f;  // should give 0.5
  input.at(1) = 2.0f;  // should give ~0.88
  input.at(2) = -2.0f; // should give ~0.12
  input.at(3) = 10.0f; // should give ~1.0

  // Test sigmoid_inline
  Tensor<float> inline_result = input;
  inline_result.set_name("inline_result");
  func::sigmoid_inline(inline_result);

  EXPECT_NEAR(inline_result.at(0), 0.5f, 1e-6);
  EXPECT_NEAR(inline_result.at(1), 0.8807970f, 1e-6);
  EXPECT_NEAR(inline_result.at(2), 0.1192029f, 1e-6);
  EXPECT_NEAR(inline_result.at(3), 0.9999546f, 1e-6);

  // Test sigmoid_out
  Tensor<float> out_result("out_result", {2, 2});
  func::sigmoid_out(input, out_result);

  EXPECT_NEAR(out_result.at(0), 0.5f, 1e-6);
  EXPECT_NEAR(out_result.at(1), 0.8807970f, 1e-6);
  EXPECT_NEAR(out_result.at(2), 0.1192029f, 1e-6);
  EXPECT_NEAR(out_result.at(3), 0.9999546f, 1e-6);
}

// Test silu tensor operations
TEST(ActivationTest, SiluTensor) {
  // Create input tensor
  Tensor<float> input("input", {2, 2});
  input.at(0) = 0.0f;
  input.at(1) = 2.0f;
  input.at(2) = -2.0f;
  input.at(3) = 4.0f;

  // Test silu_inline
  Tensor<float> inline_result = input;
  inline_result.set_name("inline_result");
  func::silu_inline(inline_result);

  // Verify results
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(inline_result.at(i), func::silu(input.at(i)), 1e-6);
  }

  // Test silu_out
  Tensor<float> out_result("out_result", {2, 2});
  func::silu_out(input, out_result);

  // Verify results
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(out_result.at(i), func::silu(input.at(i)), 1e-6);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}