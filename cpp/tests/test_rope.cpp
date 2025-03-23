#include "../src/rope.hpp"
#include "../src/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

class RoPETest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup code if needed
  }
};

// Helper function to check if two floating point numbers are approximately
// equal
template <typename T> bool approx_equal(T a, T b, T epsilon = 1e-6) {
  return std::abs(a - b) < epsilon;
}

TEST_F(RoPETest, InitCisBasicTest) {
  const size_t head_dim = 8;
  const size_t max_seq_len = 4;

  Tensor<float> cos("cos");
  Tensor<float> sin("sin");

  // Initialize cos and sin tensors
  func::init_cis(cos, sin, max_seq_len, head_dim);

  // Test basic properties
  ASSERT_EQ(cos.shape(0), head_dim / 2);
  ASSERT_EQ(cos.shape(1), max_seq_len);
  ASSERT_EQ(sin.shape(0), head_dim / 2);
  ASSERT_EQ(sin.shape(1), max_seq_len);

  // Test that values are in valid range [-1, 1]
  for (size_t pos = 0; pos < max_seq_len; ++pos) {
    for (size_t d = 0; d < head_dim / 2; ++d) {
      ASSERT_TRUE(cos.at(d, pos) >= -1.0f && cos.at(d, pos) <= 1.0f);
      ASSERT_TRUE(sin.at(d, pos) >= -1.0f && sin.at(d, pos) <= 1.0f);
    }
  }

  // Test that cos²(x) + sin²(x) ≈ 1
  for (size_t pos = 0; pos < max_seq_len; ++pos) {
    for (size_t d = 0; d < head_dim / 2; ++d) {
      float sum_squares =
          cos.at(d, pos) * cos.at(d, pos) + sin.at(d, pos) * sin.at(d, pos);
      ASSERT_TRUE(approx_equal(sum_squares, 1.0f));
    }
  }
}

TEST_F(RoPETest, RopeInlineBasicTest) {
  const size_t head_dim = 4;
  const size_t num_heads = 2;
  const size_t seq_len = 3;
  const size_t batch_size = 1;

  // Create input tensor and initialize with known values
  Tensor<float> qk("qk", {head_dim, num_heads, seq_len, batch_size});
  Tensor<float> cos("cos");
  Tensor<float> sin("sin");

  qk.linspace(1.0f, 0.0f);

  // Initialize rotation matrices
  func::init_cis(cos, sin, seq_len, head_dim);

  // Apply RoPE transformation
  func::rope_inline(qk, cos, sin);

  // Test that output has expected shape
  ASSERT_EQ(qk.shape(0), head_dim);
  ASSERT_EQ(qk.shape(1), num_heads);
  ASSERT_EQ(qk.shape(2), seq_len);
  ASSERT_EQ(qk.shape(3), batch_size);

  // Test that values have changed from initial values
  bool has_different_values = false;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t l = 0; l < seq_len; ++l) {
      for (size_t n = 0; n < num_heads; ++n) {
        for (size_t d = 0; d < head_dim; ++d) {
          if (!approx_equal(qk.at(d, n, l, b), 1.0f)) {
            has_different_values = true;
            break;
          }
        }
      }
    }
  }
  ASSERT_TRUE(has_different_values);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}