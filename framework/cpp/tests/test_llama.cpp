#include <gtest/gtest.h>

#include "llama/llama.hpp"
#include "logger.hpp"

TEST(test_tinystories260k, basic) {
  auto name = "tinystories260k";
  auto logger = get_logger();
  logger->info("Initialize model: {}", name);

  Tensor<float> token;
  auto model = llama::make_fp32_llama(name, token);
  logger->info(model.str());
}

int main(int argc, char **argv) {
  setup_logger("default");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}