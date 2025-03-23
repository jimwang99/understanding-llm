#include "../src/logger.hpp"
#include "../src/matmul.hpp"
#include "../src/tensor.hpp"
#include <gtest/gtest.h>

class MatmulTest : public ::testing::Test {
protected:
  LoggerPtr logger_;

  void SetUp() override {
    logger_ = setup_logger();
    MINFO("SetUp start");
  }
  void TearDown() override { MINFO("TearDown"); }
};

// Test basic matrix multiplication
TEST_F(MatmulTest, BasicMatmul) {
  // Create input tensors
  Tensor<float> x("x", {2, 3, 1});
  Tensor<float> y("y", {4, 2, 1});
  Tensor<float> z("z", {4, 3, 1});
  x.linspace(1.0f, 1.0f);
  y.linspace(1.0f, 1.0f);
  z.zeros();

  TRACE("x={}", x.str());
  TRACE("y={}", y.str());

  // Perform matrix multiplication
  func::matmul_2d_out<float>(x, y, z);

  // Expected result:
  // [1 2]   [1 2 3 4]   [11 14 17 20]
  // [3 4] * [5 6 7 8] = [23 30 37 44]
  // [5 6]               [35 46 57 68]
  EXPECT_FLOAT_EQ(z.at(0, 0, 0), 11);
  EXPECT_FLOAT_EQ(z.at(1, 0, 0), 14);
  EXPECT_FLOAT_EQ(z.at(2, 0, 0), 17);
  EXPECT_FLOAT_EQ(z.at(3, 0, 0), 20);

  EXPECT_FLOAT_EQ(z.at(0, 1, 0), 23);
  EXPECT_FLOAT_EQ(z.at(1, 1, 0), 30);
  EXPECT_FLOAT_EQ(z.at(2, 1, 0), 37);
  EXPECT_FLOAT_EQ(z.at(3, 1, 0), 44);

  EXPECT_FLOAT_EQ(z.at(0, 2, 0), 35);
  EXPECT_FLOAT_EQ(z.at(1, 2, 0), 46);
  EXPECT_FLOAT_EQ(z.at(2, 2, 0), 57);
  EXPECT_FLOAT_EQ(z.at(3, 2, 0), 68);

  TRACE("z={}", z.str());
}

// Test matrix multiplication with implicit transpose
TEST_F(MatmulTest, ImplicitTranspose) {
  // Create input tensors
  Tensor<float> x("x", {2, 3, 1});
  Tensor<float> y("y", {2, 4, 1});
  Tensor<float> z("z", {4, 3, 1});

  x.linspace(1.0f, 1.0f);
  y.linspace(1.0f, 1.0f);
  z.zeros();

  TRACE("x={}", x.str());
  TRACE("y={}", y.str());

  // Perform matrix multiplication with implicit transpose
  func::matmul_2d_out<float, true>(x, y, z);

  // Expected result:
  // [1 2]   [1 3 5 7]   [ 5 11 17 23]
  // [3 4] * [2 4 6 8] = [11 25 39 53]
  // [5 6]               [17 39 61 83]
  EXPECT_FLOAT_EQ(z.at(0, 0, 0), 5);
  EXPECT_FLOAT_EQ(z.at(1, 0, 0), 11);
  EXPECT_FLOAT_EQ(z.at(2, 0, 0), 17);
  EXPECT_FLOAT_EQ(z.at(3, 0, 0), 23);

  EXPECT_FLOAT_EQ(z.at(0, 1, 0), 11);
  EXPECT_FLOAT_EQ(z.at(1, 1, 0), 25);
  EXPECT_FLOAT_EQ(z.at(2, 1, 0), 39);
  EXPECT_FLOAT_EQ(z.at(3, 1, 0), 53);

  EXPECT_FLOAT_EQ(z.at(0, 2, 0), 17);
  EXPECT_FLOAT_EQ(z.at(1, 2, 0), 39);
  EXPECT_FLOAT_EQ(z.at(2, 2, 0), 61);
  EXPECT_FLOAT_EQ(z.at(3, 2, 0), 83);

  TRACE("z={}", z.str());
}

// Test batch matrix multiplication
TEST_F(MatmulTest, BatchMatmul) {
  // Create input tensors with batch size 2
  Tensor<float> x("x", {2, 2, 2});
  Tensor<float> y("y", {2, 2, 2});
  Tensor<float> z("z", {2, 2, 2});

  x.linspace(1.0f, 1.0f);
  y.linspace(1.0f, 1.0f);
  z.zeros();

  TRACE("x={}", x.str());
  TRACE("y={}", y.str());

  // Perform batch matrix multiplication
  func::matmul_2d_out<float>(x, y, z);

  // Check first batch result
  // [1 2]   [1 2]   [ 7 10]
  // [3 4] * [3 4] = [15 22]
  EXPECT_FLOAT_EQ(z.at(0, 0, 0), 7);
  EXPECT_FLOAT_EQ(z.at(1, 0, 0), 10);
  EXPECT_FLOAT_EQ(z.at(0, 1, 0), 15);
  EXPECT_FLOAT_EQ(z.at(1, 1, 0), 22);

  // Check second batch result
  // [5 6]   [5 6]   [67 78]
  // [7 8] * [7 8] = [91 106]
  EXPECT_FLOAT_EQ(z.at(0, 0, 1), 67);
  EXPECT_FLOAT_EQ(z.at(1, 0, 1), 78);
  EXPECT_FLOAT_EQ(z.at(0, 1, 1), 91);
  EXPECT_FLOAT_EQ(z.at(1, 1, 1), 106);

  TRACE("z={}", z.str());
}

// Test edge cases
TEST_F(MatmulTest, EdgeCases) {
  // 1x1 matrices
  Tensor<float> x1("x1", {1, 1, 1});
  Tensor<float> y1("y1", {1, 1, 1});
  Tensor<float> z1("z1", {1, 1, 1});
  x1.at(0) = 2.0f;
  y1.at(0) = 3.0f;
  func::matmul_2d_out<float>(x1, y1, z1);
  EXPECT_FLOAT_EQ(z1.at(0), 6.0f);

  // 1xN * Nx1 matrices
  Tensor<float> x2("x2", {3, 1, 1});
  Tensor<float> y2("y2", {1, 3, 1});
  Tensor<float> z2("z2", {1, 1, 1});
  x2.linspace(1.0f, 1.0f);
  y2.linspace(1.0f, 1.0f);
  func::matmul_2d_out<float>(x2, y2, z2);
  EXPECT_FLOAT_EQ(z2.at(0), 14.0f);
}

// Test error cases (using death tests)
TEST_F(MatmulTest, ErrorCases) {
  // Incompatible dimensions
  Tensor<float> x1("x", {2, 3, 1});
  Tensor<float> y1("y", {5, 10, 1});
  Tensor<float> z1("z", {5, 3, 1});

  EXPECT_DEATH(func::matmul_2d_out<float>(x1, y1, z1), ".*");

  Tensor<float> x2("x", {2, 3, 1});
  Tensor<float> y2("y", {5, 2, 1});
  Tensor<float> z2("z", {5, 10, 1});

  EXPECT_DEATH(func::matmul_2d_out<float>(x2, y2, z2), ".*");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}