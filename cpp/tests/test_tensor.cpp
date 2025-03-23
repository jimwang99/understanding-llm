#include "../src/logger.hpp"
#include "../src/tensor.hpp"
#include <gtest/gtest.h>
#include <sstream>

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test constructors
TEST_F(TensorTest, DefaultConstructor) {
  Tensor<float> tensor("test-tensor");
  EXPECT_EQ(tensor.name(), "test-tensor");
  EXPECT_EQ(tensor.size(), 1);
  EXPECT_EQ(tensor.ndim(), 1);
}

TEST_F(TensorTest, ShapeConstructor) {
  Tensor<float> tensor("shape-tensor", {2, 3});
  EXPECT_EQ(tensor.name(), "shape-tensor");
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);
}

TEST_F(TensorTest, ValueConstructor) {
  Tensor<int> tensor("value-tensor", {2, 3}, {1, 2, 3, 4, 5, 6});
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
  Tensor<float> tensor("view-tensor", {6});
  tensor.linspace(0.0f, 1.0f);

  tensor.view({2, 3});
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
  Tensor<float> tensor("reshape-tensor", {2, 3});
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);

  tensor.reshape({2, 4});
  EXPECT_EQ(tensor.size(), 8);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 4);

  tensor.reshape({10});
  EXPECT_EQ(tensor.size(), 10);
  EXPECT_EQ(tensor.ndim(), 1);
  EXPECT_EQ(tensor.shape(0), 10);
}

TEST_F(TensorTest, SetName) {
  Tensor<float> tensor("set-name-tensor");
  EXPECT_EQ(tensor.name(), "set-name-tensor");

  tensor.set_name("new-name");
  EXPECT_EQ(tensor.name(), "new-name");
}

// Test initializers
TEST_F(TensorTest, Zeros) {
  Tensor<float> tensor("zeros-tensor", {2, 3});

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
  Tensor<float> tensor("linspace-tensor", {5});

  tensor.linspace(1.0f, 2.0f);
  EXPECT_EQ(tensor.at(0), 1.0f);
  EXPECT_EQ(tensor.at(1), 3.0f);
  EXPECT_EQ(tensor.at(2), 5.0f);
  EXPECT_EQ(tensor.at(3), 7.0f);
  EXPECT_EQ(tensor.at(4), 9.0f);
}

// Test accessors
TEST_F(TensorTest, AccessorsOneDim) {
  Tensor<int> tensor("accessors-tensor", {5}, {10, 20, 30, 40, 50});

  EXPECT_EQ(tensor.at(0), 10);
  EXPECT_EQ(tensor.at(4), 50);

  tensor.at(2) = 35;
  EXPECT_EQ(tensor.at(2), 35);
}

TEST_F(TensorTest, AccessorsTwoDim) {
  Tensor<int> tensor("accessors-tensor", {2, 3}, {1, 2, 3, 4, 5, 6});

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
  Tensor<int> tensor("accessors-tensor", {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

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
  Tensor<int> tensor("accessors-tensor", {2, 2, 2, 2},
                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
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
  Tensor<int> tensor("data-accessors-tensor", {3}, {1, 2, 3});

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
  Tensor<float> tensor("properties-tensor", {2, 3, 4});

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
  Tensor<int> t1("comparison-tensor1", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor<int> t2("comparison-tensor2", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor<int> t3("comparison-tensor3", {2, 3}, {1, 2, 3, 4, 5, 7});
  Tensor<int> t4("comparison-tensor4", {3, 2}, {1, 2, 3, 4, 5, 6});

  EXPECT_TRUE(t1 == t2);
  EXPECT_FALSE(t1 == t3);
  EXPECT_FALSE(t1 == t4);
}

// Test ostream operator
TEST_F(TensorTest, OstreamOperator) {
  Tensor<int> tensor("test-tensor", {3}, {1, 2, 3});

  std::stringstream ss;
  ss << tensor;

  EXPECT_TRUE(ss.str().find("test-tensor") != std::string::npos);
  EXPECT_TRUE(ss.str().find("1, 2, 3") != std::string::npos);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
