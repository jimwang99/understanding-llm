#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

TEST(test_spdlog, basic) {
  spdlog::critical("Critical message");
  spdlog::error("Error message");
  spdlog::warn("Warning message");
  spdlog::info("Info message");
  spdlog::debug("Debug message");
}

TEST(test_env, basic) {
  EXPECT_TRUE(1 == 1);
  EXPECT_FALSE(1 == 2);
  EXPECT_EQ(1 + 1, 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}