#include "../src/logger.hpp"
#include "../src/tensor.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <sstream>

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Common setup code
    logger = setup_logger();
  }

  void TearDown() override {
    // Common cleanup code
    spdlog::drop_all();
  }

  logger_t logger;
};

// Test constructors
TEST_F(TensorTest, DefaultConstructor) {
  Tensor<float> tensor("test-tensor");
  EXPECT_EQ(tensor.name(), "test-tensor");
  EXPECT_EQ(tensor.size(), 1);
  EXPECT_EQ(tensor.ndim(), 1);
}

TEST_F(TensorTest, ShapeConstructor) {
  std::vector<size_t> shape = {2, 3};
  Tensor<float> tensor(shape, "shape-tensor");
  EXPECT_EQ(tensor.name(), "shape-tensor");
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);
}

TEST_F(TensorTest, ValueConstructor) {
  std::vector<size_t> shape = {2, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6};
  Tensor<int> tensor(shape, values, "value-tensor");
  EXPECT_EQ(tensor.name(), "value-tensor");
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);

  EXPECT_EQ(tensor.at(0), 1);
  EXPECT_EQ(tensor.at(1), 2);
  EXPECT_EQ(tensor.at(2), 3);
  EXPECT_EQ(tensor.at(3), 4);
  EXPECT_EQ(tensor.at(4), 5);
  EXPECT_EQ(tensor.at(5), 6);
  EXPECT_EQ(tensor.at(0, 0), 1);
  EXPECT_EQ(tensor.at(1, 0), 2);
  EXPECT_EQ(tensor.at(0, 1), 3);
  EXPECT_EQ(tensor.at(1, 1), 4);
  EXPECT_EQ(tensor.at(0, 2), 5);
  EXPECT_EQ(tensor.at(1, 2), 6);
}

TEST_F(TensorTest, View) {
  std::vector<size_t> shape1 = {6};
  Tensor<float> tensor(shape1);
  tensor.linspace(0.0f, 1.0f);

  std::vector<size_t> shape2 = {2, 3};
  tensor.view(shape2);
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);

  EXPECT_EQ(tensor.at(0, 0), 0.0f);
  EXPECT_EQ(tensor.at(1, 0), 1.0f);
  EXPECT_EQ(tensor.at(0, 1), 2.0f);
  EXPECT_EQ(tensor.at(1, 1), 3.0f);
  EXPECT_EQ(tensor.at(0, 2), 4.0f);
  EXPECT_EQ(tensor.at(1, 2), 5.0f);
}

// Test modifiers
TEST_F(TensorTest, Reshape) {
  std::vector<size_t> shape1 = {2, 3};
  Tensor<float> tensor(shape1);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);

  std::vector<size_t> shape2 = {2, 4};
  tensor.reshape(shape2);
  EXPECT_EQ(tensor.size(), 8);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 4);

  std::vector<size_t> shape3 = {10};
  tensor.reshape(shape3);
  EXPECT_EQ(tensor.size(), 10);
  EXPECT_EQ(tensor.ndim(), 1);
  EXPECT_EQ(tensor.shape(0), 10);
}

TEST_F(TensorTest, SetName) {
  Tensor<float> tensor;
  EXPECT_EQ(tensor.name(), "Unnamed-Tensor");

  tensor.set_name("new-name");
  EXPECT_EQ(tensor.name(), "new-name");
}

// Test initializers
TEST_F(TensorTest, Zeros) {
  std::vector<size_t> shape = {2, 3};
  Tensor<float> tensor(shape);

  // Fill with some values first
  for (size_t i = 0; i < tensor.size(); i++) {
    tensor.at(i) = 1.0f;
  }

  tensor.zeros();
  for (size_t i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.at(i), 0.0f);
  }
}

TEST_F(TensorTest, Linspace) {
  std::vector<size_t> shape = {5};
  Tensor<float> tensor(shape);

  tensor.linspace(1.0f, 2.0f);
  EXPECT_EQ(tensor.at(0), 1.0f);
  EXPECT_EQ(tensor.at(1), 3.0f);
  EXPECT_EQ(tensor.at(2), 5.0f);
  EXPECT_EQ(tensor.at(3), 7.0f);
  EXPECT_EQ(tensor.at(4), 9.0f);
}

// Test accessors
TEST_F(TensorTest, AccessorsOneDim) {
  std::vector<size_t> shape = {5};
  std::vector<int> values = {10, 20, 30, 40, 50};
  Tensor<int> tensor(shape, values);

  EXPECT_EQ(tensor.at(0), 10);
  EXPECT_EQ(tensor.at(4), 50);

  tensor.at(2) = 35;
  EXPECT_EQ(tensor.at(2), 35);
}

TEST_F(TensorTest, AccessorsTwoDim) {
  std::vector<size_t> shape = {2, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6};
  Tensor<int> tensor(shape, values);

  EXPECT_EQ(tensor.at(0, 0), 1);
  EXPECT_EQ(tensor.at(1, 0), 2);
  EXPECT_EQ(tensor.at(0, 1), 3);
  EXPECT_EQ(tensor.at(1, 1), 4);
  EXPECT_EQ(tensor.at(0, 2), 5);
  EXPECT_EQ(tensor.at(1, 2), 6);

  tensor.at(1, 1) = 55;
  EXPECT_EQ(tensor.at(1, 1), 55);
}

TEST_F(TensorTest, AccessorsThreeDim) {
  std::vector<size_t> shape = {2, 2, 2};
  Tensor<int> tensor(shape);
  for (size_t i = 0; i < tensor.size(); i++) {
    tensor.at(i) = static_cast<int>(i + 1);
  }

  EXPECT_EQ(tensor.at(0, 0, 0), 1);
  EXPECT_EQ(tensor.at(1, 0, 0), 2);
  EXPECT_EQ(tensor.at(0, 1, 0), 3);
  EXPECT_EQ(tensor.at(1, 1, 0), 4);
  EXPECT_EQ(tensor.at(0, 0, 1), 5);
  EXPECT_EQ(tensor.at(1, 0, 1), 6);
  EXPECT_EQ(tensor.at(0, 1, 1), 7);
  EXPECT_EQ(tensor.at(1, 1, 1), 8);

  tensor.at(1, 0, 1) = 66;
  EXPECT_EQ(tensor.at(1, 0, 1), 66);
}

TEST_F(TensorTest, AccessorsFourDim) {
  std::vector<size_t> shape = {2, 2, 2, 2};
  Tensor<int> tensor(shape);
  for (size_t i = 0; i < tensor.size(); i++) {
    tensor.at(i) = static_cast<int>(i + 1);
  }

  EXPECT_EQ(tensor.at(0, 0, 0, 0), 1);
  EXPECT_EQ(tensor.at(1, 1, 1, 1), 16);

  tensor.at(0, 1, 0, 1) = 99;
  EXPECT_EQ(tensor.at(0, 1, 0, 1), 99);
}

// Test data accessors
TEST_F(TensorTest, DataAccessors) {
  std::vector<size_t> shape = {3};
  std::vector<int> values = {1, 2, 3};
  Tensor<int> tensor(shape, values);

  const int *data = tensor.data();
  EXPECT_EQ(data[0], 1);
  EXPECT_EQ(data[1], 2);
  EXPECT_EQ(data[2], 3);

  int *mutable_data = tensor.mutable_data();
  mutable_data[1] = 22;
  EXPECT_EQ(tensor.at(1), 22);
}

// Test properties
TEST_F(TensorTest, Properties) {
  std::vector<size_t> shape = {2, 3, 4};
  Tensor<float> tensor(shape);

  EXPECT_EQ(tensor.size(), 24);
  EXPECT_EQ(tensor.nbytes(), 24 * sizeof(float));
  EXPECT_EQ(tensor.ndim(), 3);

  std::vector<size_t> expected_shape = {2, 3, 4};
  EXPECT_EQ(tensor.shape(), expected_shape);

  std::vector<size_t> expected_strides = {1, 2, 6, 24};
  EXPECT_EQ(tensor.stride(), expected_strides);
}

// Test comparison operators
TEST_F(TensorTest, ComparisonOperators) {
  std::vector<size_t> shape1 = {2, 3};
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6};
  Tensor<int> t1(shape1, values1);

  std::vector<size_t> shape2 = {2, 3};
  std::vector<int> values2 = {1, 2, 3, 4, 5, 6};
  Tensor<int> t2(shape2, values2);

  std::vector<size_t> shape3 = {2, 3};
  std::vector<int> values3 = {1, 2, 3, 4, 5, 7}; // Last element different
  Tensor<int> t3(shape3, values3);

  std::vector<size_t> shape4 = {3, 2};
  std::vector<int> values4 = {1, 2, 3, 4, 5, 6};
  Tensor<int> t4(shape4, values4);

  EXPECT_TRUE(t1 == t2);
  EXPECT_FALSE(t1 == t3);
  EXPECT_FALSE(t1 == t4);
}

// Test ostream operator
TEST_F(TensorTest, OstreamOperator) {
  std::vector<size_t> shape = {3};
  std::vector<int> values = {1, 2, 3};
  Tensor<int> tensor(shape, values, "test-tensor");

  std::stringstream ss;
  ss << tensor;

  EXPECT_TRUE(ss.str().find("test-tensor") != std::string::npos);
  EXPECT_TRUE(ss.str().find("1, 2, 3") != std::string::npos);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
