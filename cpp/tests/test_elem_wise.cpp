#include "../src/elem_wise.hpp"
#include "../src/tensor.hpp"
#include <gtest/gtest.h>

class ElemWiseTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup code if needed
  }
};

// Test regular element-wise operations
TEST_F(ElemWiseTest, AddInline) {
  Tensor<float> x("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> y("y", {2, 2}, {0.5f, 1.5f, 2.5f, 3.5f});

  func::add_inline(x, y);

  EXPECT_FLOAT_EQ(x.at(0), 1.5f);
  EXPECT_FLOAT_EQ(x.at(1), 3.5f);
  EXPECT_FLOAT_EQ(x.at(2), 5.5f);
  EXPECT_FLOAT_EQ(x.at(3), 7.5f);
}

TEST_F(ElemWiseTest, SubInline) {
  Tensor<float> x("x", {2, 2}, {4.0f, 3.0f, 2.0f, 1.0f});
  Tensor<float> y("y", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});

  func::sub_inline(x, y);

  EXPECT_FLOAT_EQ(x.at(0), 3.0f);
  EXPECT_FLOAT_EQ(x.at(1), 2.0f);
  EXPECT_FLOAT_EQ(x.at(2), 1.0f);
  EXPECT_FLOAT_EQ(x.at(3), 0.0f);
}

TEST_F(ElemWiseTest, MulInline) {
  Tensor<float> x("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> y("y", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});

  func::mul_inline(x, y);

  EXPECT_FLOAT_EQ(x.at(0), 2.0f);
  EXPECT_FLOAT_EQ(x.at(1), 4.0f);
  EXPECT_FLOAT_EQ(x.at(2), 6.0f);
  EXPECT_FLOAT_EQ(x.at(3), 8.0f);
}

TEST_F(ElemWiseTest, DivInline) {
  Tensor<float> x("x", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  Tensor<float> y("y", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});

  func::div_inline(x, y);

  EXPECT_FLOAT_EQ(x.at(0), 1.0f);
  EXPECT_FLOAT_EQ(x.at(1), 2.0f);
  EXPECT_FLOAT_EQ(x.at(2), 3.0f);
  EXPECT_FLOAT_EQ(x.at(3), 4.0f);
}

// Test scalar operations
TEST_F(ElemWiseTest, AddScalarInline) {
  Tensor<float> x("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  func::add_scalar_inline(x, 2.0f);

  EXPECT_FLOAT_EQ(x.at(0), 3.0f);
  EXPECT_FLOAT_EQ(x.at(1), 4.0f);
  EXPECT_FLOAT_EQ(x.at(2), 5.0f);
  EXPECT_FLOAT_EQ(x.at(3), 6.0f);
}

TEST_F(ElemWiseTest, SubScalarInline) {
  Tensor<float> x("x", {2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
  func::sub_scalar_inline(x, 1.0f);

  EXPECT_FLOAT_EQ(x.at(0), 2.0f);
  EXPECT_FLOAT_EQ(x.at(1), 3.0f);
  EXPECT_FLOAT_EQ(x.at(2), 4.0f);
  EXPECT_FLOAT_EQ(x.at(3), 5.0f);
}

TEST_F(ElemWiseTest, MulScalarInline) {
  Tensor<float> x("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  func::mul_scalar_inline(x, 3.0f);

  EXPECT_FLOAT_EQ(x.at(0), 3.0f);
  EXPECT_FLOAT_EQ(x.at(1), 6.0f);
  EXPECT_FLOAT_EQ(x.at(2), 9.0f);
  EXPECT_FLOAT_EQ(x.at(3), 12.0f);
}

TEST_F(ElemWiseTest, DivScalarInline) {
  Tensor<float> x("x", {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  func::div_scalar_inline(x, 2.0f);

  EXPECT_FLOAT_EQ(x.at(0), 1.0f);
  EXPECT_FLOAT_EQ(x.at(1), 2.0f);
  EXPECT_FLOAT_EQ(x.at(2), 3.0f);
  EXPECT_FLOAT_EQ(x.at(3), 4.0f);
}

// Test broadcasting operations
TEST_F(ElemWiseTest, AddBroadcastInline) {
  Tensor<float> matrix("matrix", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor<float> vector("vector", {3}, {0.5f, 1.0f, 1.5f});

  func::add_broadcast_inline(matrix, vector);

  EXPECT_FLOAT_EQ(matrix.at(0, 0), 1.5f);
  EXPECT_FLOAT_EQ(matrix.at(1, 0), 3.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 0), 4.5f);
  EXPECT_FLOAT_EQ(matrix.at(0, 1), 4.5f);
  EXPECT_FLOAT_EQ(matrix.at(1, 1), 6.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 1), 7.5f);
}

TEST_F(ElemWiseTest, SubBroadcastInline) {
  Tensor<float> matrix("matrix", {3, 2}, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
  Tensor<float> vector("vector", {3}, {1.0f, 2.0f, 3.0f});

  func::sub_broadcast_inline(matrix, vector);

  EXPECT_FLOAT_EQ(matrix.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix.at(0, 1), 4.0f);
  EXPECT_FLOAT_EQ(matrix.at(1, 1), 4.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 1), 4.0f);
}

TEST_F(ElemWiseTest, MulBroadcastInline) {
  Tensor<float> matrix("matrix", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor<float> vector("vector", {3}, {0.5f, 1.0f, 1.5f});

  func::mul_broadcast_inline(matrix, vector);

  EXPECT_FLOAT_EQ(matrix.at(0, 0), 0.5f);
  EXPECT_FLOAT_EQ(matrix.at(1, 0), 2.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 0), 4.5f);
  EXPECT_FLOAT_EQ(matrix.at(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(matrix.at(1, 1), 5.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 1), 9.0f);
}

TEST_F(ElemWiseTest, DivBroadcastInline) {
  Tensor<float> matrix("matrix", {3, 2}, {2.0f, 4.0f, 8.0f, 4.0f, 8.0f, 16.0f});
  Tensor<float> vector("vector", {3}, {2.0f, 4.0f, 8.0f});

  func::div_broadcast_inline(matrix, vector);

  EXPECT_FLOAT_EQ(matrix.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 0), 1.0f);
  EXPECT_FLOAT_EQ(matrix.at(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(matrix.at(1, 1), 2.0f);
  EXPECT_FLOAT_EQ(matrix.at(2, 1), 2.0f);
}

// Test error cases
TEST_F(ElemWiseTest, InlineOperationErrors) {
  // Test mismatched shapes - different dimensions
  Tensor<float> x1("x1", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> y1("y1", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_DEATH(func::add_inline(x1, y1), ".*");
  EXPECT_DEATH(func::sub_inline(x1, y1), ".*");
  EXPECT_DEATH(func::mul_inline(x1, y1), ".*");
  EXPECT_DEATH(func::div_inline(x1, y1), ".*");

  // Test mismatched shapes - same dimensions, different sizes
  Tensor<float> x2("x2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> y2("y2", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  EXPECT_DEATH(func::add_inline(x2, y2), ".*");
  EXPECT_DEATH(func::sub_inline(x2, y2), ".*");
  EXPECT_DEATH(func::mul_inline(x2, y2), ".*");
  EXPECT_DEATH(func::div_inline(x2, y2), ".*");

  // Test mismatched shapes - different number of elements
  Tensor<float> x3("x3", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> y3("y3", {2, 1}, {1.0f, 2.0f});
  EXPECT_DEATH(func::add_inline(x3, y3), ".*");
  EXPECT_DEATH(func::sub_inline(x3, y3), ".*");
  EXPECT_DEATH(func::mul_inline(x3, y3), ".*");
  EXPECT_DEATH(func::div_inline(x3, y3), ".*");
}

TEST_F(ElemWiseTest, BroadcastOperationErrors) {
  // Test invalid dimensions (matrix should be 2D, vector should be 1D)
  Tensor<float> matrix1("matrix1", {2, 2, 2},
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  Tensor<float> vector1("vector1", {2}, {1.0f, 2.0f});
  EXPECT_DEATH(func::add_broadcast_inline(matrix1, vector1), ".*");
  EXPECT_DEATH(func::sub_broadcast_inline(matrix1, vector1), ".*");
  EXPECT_DEATH(func::mul_broadcast_inline(matrix1, vector1), ".*");
  EXPECT_DEATH(func::div_broadcast_inline(matrix1, vector1), ".*");

  // Test vector with wrong dimension
  Tensor<float> matrix2("matrix2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> vector2("vector2", {2, 1}, {1.0f, 2.0f});
  EXPECT_DEATH(func::add_broadcast_inline(matrix2, vector2), ".*");
  EXPECT_DEATH(func::sub_broadcast_inline(matrix2, vector2), ".*");
  EXPECT_DEATH(func::mul_broadcast_inline(matrix2, vector2), ".*");
  EXPECT_DEATH(func::div_broadcast_inline(matrix2, vector2), ".*");

  // Test mismatched sizes (vector size should match matrix's first dimension)
  Tensor<float> matrix3("matrix3", {2, 3},
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor<float> vector3("vector3", {3}, {1.0f, 2.0f, 3.0f});
  EXPECT_DEATH(func::add_broadcast_inline(matrix3, vector3), ".*");
  EXPECT_DEATH(func::sub_broadcast_inline(matrix3, vector3), ".*");
  EXPECT_DEATH(func::mul_broadcast_inline(matrix3, vector3), ".*");
  EXPECT_DEATH(func::div_broadcast_inline(matrix3, vector3), ".*");
}

TEST_F(ElemWiseTest, DivisionByZeroErrors) {
  // Test inline division by zero
  Tensor<float> x1("x1", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> y1("y1", {2, 2}, {0.0f, 1.0f, 0.0f, 1.0f});
  EXPECT_DEATH(func::div_inline(x1, y1), ".* != 0.*");

  // Test scalar division by zero
  Tensor<float> x2("x2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_DEATH(func::div_scalar_inline(x2, 0.0f), ".* != 0.*");

  // Test broadcast division by zero
  Tensor<float> matrix("matrix", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor<float> vector("vector", {3}, {0.0f, 1.0f, 0.0f});
  EXPECT_DEATH(func::div_broadcast_inline(matrix, vector), ".* != 0.*");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
