#include "../src/tensor.hpp"
#include "../src/transpose.hpp"
#include <gtest/gtest.h>

class TransposeTest : public ::testing::Test {
protected:
  void SetUp() override { get_logger(); }
};

TEST_F(TransposeTest, Basic3DTranspose) {
  // Create a 2x3x2 tensor
  Tensor<float> x("input", {1, 2, 3});
  x.linspace(1.0f, 1.0f);

  TRACE("before transpose: {}", x.str());
  func::tranpose_1_2_inline(x);
  TRACE("after transpose: {}", x.str());

  EXPECT_EQ(x.shape()[0], 1u);
  EXPECT_EQ(x.shape()[1], 3u);
  EXPECT_EQ(x.shape()[2], 2u);

  EXPECT_FLOAT_EQ(x.at(0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(x.at(0, 1, 0), 3.0f);
  EXPECT_FLOAT_EQ(x.at(0, 2, 0), 5.0f);
  EXPECT_FLOAT_EQ(x.at(0, 0, 1), 2.0f);
  EXPECT_FLOAT_EQ(x.at(0, 1, 1), 4.0f);
  EXPECT_FLOAT_EQ(x.at(0, 2, 1), 6.0f);
}

TEST_F(TransposeTest, TransposeWithBatch) {
  Tensor<float> x("batch_input", {1, 2, 3, 4});
  x.linspace(1.0f, 1.0f);

  TRACE("before transpose: {}", x.str());
  func::tranpose_1_2_inline(x);
  TRACE("after transpose: {}", x.str());

  auto shape = x.shape();
  EXPECT_EQ(shape[0], 1u);
  EXPECT_EQ(shape[1], 3u);
  EXPECT_EQ(shape[2], 2u);
  EXPECT_EQ(shape[3], 4u);

  EXPECT_FLOAT_EQ(x.at(0), 1.0f);
  EXPECT_FLOAT_EQ(x.at(1), 3.0f);
  EXPECT_FLOAT_EQ(x.at(2), 5.0f);
  EXPECT_FLOAT_EQ(x.at(3), 2.0f);
  EXPECT_FLOAT_EQ(x.at(4), 4.0f);
  EXPECT_FLOAT_EQ(x.at(5), 6.0f);
  EXPECT_FLOAT_EQ(x.at(21), 20.0f);
  EXPECT_FLOAT_EQ(x.at(22), 22.0f);
  EXPECT_FLOAT_EQ(x.at(23), 24.0f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
