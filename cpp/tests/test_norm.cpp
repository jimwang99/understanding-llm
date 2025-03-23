#include "../src/norm.hpp"
#include "../src/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Test RMS normalization with simple 1D tensor
TEST(NormTest, RmsNorm1D) {
  // Create input tensor with known values
  Tensor<float> input("input", {4});
  input.at(0) = 1.0f;
  input.at(1) = 2.0f;
  input.at(2) = -1.0f;
  input.at(3) = 3.0f;

  // Create output tensor
  Tensor<float> output("output", {4});

  // Apply RMS normalization
  float eps = 1e-5f;
  func::rms_norm_out(input, output, eps);

  // Calculate expected values
  float sum_squares =
      1.0f * 1.0f + 2.0f * 2.0f + (-1.0f) * (-1.0f) + 3.0f * 3.0f;
  float rms = std::sqrt(sum_squares / 4 + eps);

  // Verify results
  EXPECT_NEAR(output.at(0), 1.0f / rms, 1e-6);
  EXPECT_NEAR(output.at(1), 2.0f / rms, 1e-6);
  EXPECT_NEAR(output.at(2), -1.0f / rms, 1e-6);
  EXPECT_NEAR(output.at(3), 3.0f / rms, 1e-6);
}

// Test RMS normalization with 2D tensor (batch processing)
TEST(NormTest, RmsNorm2D) {
  // Create input tensor with 2 batches
  Tensor<float> input("input", {3, 2}); // 3 features, 2 batches
  // First batch
  input.at(0, 0) = 1.0f;
  input.at(1, 0) = 2.0f;
  input.at(2, 0) = 3.0f;
  // Second batch
  input.at(0, 1) = -1.0f;
  input.at(1, 1) = 0.0f;
  input.at(2, 1) = 1.0f;

  // Create output tensor
  Tensor<float> output("output", {3, 2});

  // Apply RMS normalization
  float eps = 1e-5f;
  func::rms_norm_out(input, output, eps);

  // Calculate expected values for first batch
  float sum_squares1 = 1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f;
  float rms1 = std::sqrt(sum_squares1 / 3 + eps);

  // Calculate expected values for second batch
  float sum_squares2 = (-1.0f) * (-1.0f) + 0.0f * 0.0f + 1.0f * 1.0f;
  float rms2 = std::sqrt(sum_squares2 / 3 + eps);

  // Verify results for first batch
  EXPECT_NEAR(output.at(0, 0), 1.0f / rms1, 1e-6);
  EXPECT_NEAR(output.at(1, 0), 2.0f / rms1, 1e-6);
  EXPECT_NEAR(output.at(2, 0), 3.0f / rms1, 1e-6);

  // Verify results for second batch
  EXPECT_NEAR(output.at(0, 1), -1.0f / rms2, 1e-6);
  EXPECT_NEAR(output.at(1, 1), 0.0f / rms2, 1e-6);
  EXPECT_NEAR(output.at(2, 1), 1.0f / rms2, 1e-6);
}

// Test RMS normalization with zero tensor
TEST(NormTest, RmsNormZero) {
  // Create input tensor with zeros
  Tensor<float> input("input", {3});
  input.at(0) = 0.0f;
  input.at(1) = 0.0f;
  input.at(2) = 0.0f;

  // Create output tensor
  Tensor<float> output("output", {3});

  // Apply RMS normalization with default epsilon
  func::rms_norm_out(input, output);

  // For zero input, output should be zero divided by sqrt(eps)
  float expected = 0.0f / std::sqrt(1e-5f);
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_NEAR(output.at(i), expected, 1e-6);
  }
}

// Test RMS normalization with different epsilon values
TEST(NormTest, RmsNormEpsilon) {
  // Create input tensor
  Tensor<float> input("input", {2});
  input.at(0) = 1.0f;
  input.at(1) = -1.0f;

  // Create output tensors
  Tensor<float> output1("output1", {2});
  Tensor<float> output2("output2", {2});

  // Apply RMS normalization with different epsilon values
  float eps1 = 1e-5f;
  float eps2 = 1e-3f;
  func::rms_norm_out(input, output1, eps1);
  func::rms_norm_out(input, output2, eps2);

  // Results should be different due to different epsilon values
  EXPECT_NE(output1.at(0), output2.at(0));
  EXPECT_NE(output1.at(1), output2.at(1));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
