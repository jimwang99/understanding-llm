#include "llama/llama.hpp"
#include "logger.hpp"
#include <gtest/gtest.h>

TEST(test_tinystories260k, basic) {
  auto logger = get_logger("test_llama");
  SPDLOG_LOGGER_INFO(logger, "Initialize model");
  auto model = llama::make_fp32_llama("tinystories260k");
  SPDLOG_LOGGER_INFO(logger, "{}", model.str());
}

int main(int argc, char **argv) {
  setup_logger("test_llama", "trace");
  setup_logger("llama", "trace");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
