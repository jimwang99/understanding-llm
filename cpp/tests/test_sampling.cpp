#include "../src/sampling.hpp"
#include "../src/tensor.hpp"
#include <gtest/gtest.h>

TEST(SamplingTest, GreedySamplingBasic) {
  Tensor<float> logit("logit", {3, 2, 1});
  Tensor<size_t> token("token", {2, 1});
  logit.linspace(0.1f, 0.1f);
  token.zeros();

  func::sampling_greedy(logit, token);

  EXPECT_EQ(token.at(0, 0), 2u);
  EXPECT_EQ(token.at(1, 0), 2u);
}

TEST(SamplingTest, GreedySamplingLarger) {
  Tensor<float> logit("logit", {2, 2, 2},
                      {0.1f, 0.2f, 0.4f, 0.3f, 0.1f, 0.2f, 0.4f, 0.3f});
  Tensor<size_t> token("token", {2, 2});
  token.zeros();

  func::sampling_greedy(logit, token);

  EXPECT_EQ(token.at(0, 0), 1u);
  EXPECT_EQ(token.at(1, 0), 0u);
  EXPECT_EQ(token.at(0, 1), 1u);
  EXPECT_EQ(token.at(1, 1), 0u);
}

TEST(SamplingTest, GreedySamplingEqualValues) {
  Tensor<float> logit("logit", {2, 1, 1}, {0.5f, 0.5f});
  Tensor<size_t> token("token", {1, 1});
  token.zeros();

  func::sampling_greedy(logit, token);

  EXPECT_EQ(token.at(0, 0), 0u);
}

TEST(SamplingTest, GreedySamplingInvalidShapes) {
  Tensor<float> logit("logit", {2, 2, 2});
  Tensor<size_t> token("token", {3, 2});

  EXPECT_DEATH(func::sampling_greedy(logit, token), ".*");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}