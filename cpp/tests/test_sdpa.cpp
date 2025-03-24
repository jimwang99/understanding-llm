#include "../src/sdpa.hpp"
#include "../src/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

class SDPATest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(SDPATest, BasicCausalAttnMaskInit) {
  Tensor<float> mask("mask", {3, 3});
  func::init_causal_attn_mask(mask, 3);

  TRACE(mask.str());

  // Check dimensions
  EXPECT_EQ(mask.shape(0), 3u);
  EXPECT_EQ(mask.shape(1), 3u);

  // Check causal attention pattern
  EXPECT_EQ(mask.at(0, 0), 0.0f);
  EXPECT_TRUE(std::isinf(mask.at(1, 0)));
  EXPECT_TRUE(std::isinf(mask.at(2, 0)));
  EXPECT_EQ(mask.at(0, 1), 0.0f);
  EXPECT_EQ(mask.at(1, 1), 0.0f);
  EXPECT_TRUE(std::isinf(mask.at(2, 1)));
  EXPECT_EQ(mask.at(0, 2), 0.0f);
  EXPECT_EQ(mask.at(1, 2), 0.0f);
  EXPECT_EQ(mask.at(2, 2), 0.0f);
}

TEST_F(SDPATest, BasicSDPAOut) {
  // Test parameters
  const size_t B = 1;   // batch size
  const size_t N = 1;   // num heads
  const size_t D = 2;   // head dimension
  const size_t Lq = 3;  // query sequence length
  const size_t Lkv = 3; // key/value sequence length
  const size_t Lm = 10; // max sequence length

  // Initialize input tensors
  Tensor<float> q("Q", {D, Lq, N, B});
  Tensor<float> k("K", {D, Lkv, N, B});
  Tensor<float> v("V", {D, Lkv, N, B});
  Tensor<float> attn("attn", {Lkv, Lq, N, B});
  Tensor<float> mask("mask", {Lm, Lm});

  // Initialize qkv with simple values
  q.linspace(1.0f, 1.0f);
  k.linspace(1.0f, 1.0f);
  v.linspace(1.0f, 1.0f);

  // Initialize causal attention mask
  func::init_causal_attn_mask(mask, Lm);

  // Run SDPA
  func::sdpa_out(q, k, v, attn, mask);

  // Verify output shapes are preserved
  EXPECT_EQ(q.shape(0), D);
  EXPECT_EQ(q.shape(1), Lq);
  EXPECT_EQ(q.shape(2), N);
  EXPECT_EQ(q.shape(3), B);

  EXPECT_EQ(k.shape(0), D);
  EXPECT_EQ(k.shape(1), Lkv);
  EXPECT_EQ(k.shape(2), N);
  EXPECT_EQ(k.shape(3), B);

  EXPECT_EQ(v.shape(0), D);
  EXPECT_EQ(v.shape(1), Lkv);
  EXPECT_EQ(v.shape(2), N);
  EXPECT_EQ(v.shape(3), B);

  EXPECT_EQ(attn.shape(0), Lkv);
  EXPECT_EQ(attn.shape(1), Lq);
  EXPECT_EQ(attn.shape(2), N);
  EXPECT_EQ(attn.shape(3), B);

  Tensor<float> expected_attn("expected_attn", {Lkv, Lq, N, B},
                              {1, 0, 0, 5.0197505e-05, 0.9999498, 0,
                               3.0755743e-14, 1.7537299e-07, 0.9999999});
  Tensor<float> expected_q("expected_q", {D, Lq, N, B},
                           {1, 2, 2.9998996, 3.9998996, 5, 6});
  EXPECT_TRUE(is_close(attn, expected_attn));
  EXPECT_TRUE(is_close(q, expected_q));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
