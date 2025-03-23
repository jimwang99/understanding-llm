#include "logger.hpp"
#include "module.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

class TestAffine final : public Module<float> {
private:
  CTensorPtr<float> in_;
  TensorPtr<float> out_;
  TensorPtr<float> weight_;
  TensorPtr<float> bias_;

public:
  TestAffine(const std::string name, const size_t size, CTensorPtr<float> in,
             TensorPtr<float> out)
      : Module<float>(name), in_(in), out_(out),
        weight_(
            std::make_shared<Tensor<float>>("weight1", TensorShape({size}))),
        bias_(std::make_shared<Tensor<float>>("bias1", TensorShape({size}))) {
    MINFO("Create TestAffine");
    add_param("size", size);
    add_weight(weight_);
    add_weight(bias_);
    add_input(in_);
    add_output(out_);
    MASSERT(in_->size() == size, "Input size {} must match size {}",
            in_->size(), size);
    MASSERT(out_->size() == size, "Output size {} must match size {}",
            out_->size(), size);
  }

  ~TestAffine() override { MINFO("Destroy TestAffine"); }

  void forward() override {
    MINFO("Forward TestAffine");
    MDEBUG("before: in={}", in_->str());
    MDEBUG("before: weight={}", weight_->str());
    MDEBUG("before: bias={}", bias_->str());
    MDEBUG("before: out={}", out_->str());
    for (size_t i = 0; i < param("size"); ++i) {
      out_->at(i) = in_->at(i) * weight_->at(i) + bias_->at(i);
    }
    MDEBUG("after: in={}", in_->str());
    MDEBUG("after: weight={}", weight_->str());
    MDEBUG("after: bias={}", bias_->str());
    MDEBUG("after: out={}", out_->str());
  }
};

class TestHalf final : public Module<float> {
private:
  TensorPtr<float> inout_;

public:
  TestHalf(const std::string name, const size_t size, TensorPtr<float> inout)
      : Module<float>(name), inout_(inout) {
    MINFO("Create TestHalf");
    add_param("size", size);
    add_input(inout_);
    MASSERT(
        inout_->size() == size,
        fmt::format("Input size {} must match size {}", inout_->size(), size));
  }

  ~TestHalf() override { MINFO("Destroy TestHalf"); }

  void forward() override {
    MINFO("Forward TestHalf");
    MDEBUG("before: inout={}", inout_->str());
    for (size_t i = 0; i < inout_->size(); ++i) {
      inout_->at(i) = inout_->at(i) / 2.0f;
    }
    MDEBUG("after: inout={}", inout_->str());
  }
};

class TestTop final : public Module<float> {
private:
  TensorPtr<float> a_;
  TensorPtr<float> b_;
  ModulePtr<float> m_affine_;
  ModulePtr<float> m_half_;

public:
  TestTop()
      : Module<float>("top"),
        a_(std::make_shared<Tensor<float>>("a", TensorShape({3}))),
        b_(std::make_shared<Tensor<float>>("b", TensorShape({3}))),
        m_affine_(std::make_shared<TestAffine>("affine", 3, a_, b_)),
        m_half_(std::make_shared<TestHalf>("half", 3, b_)) {
    MINFO("Create TestTop");
    add_submodule(m_affine_);
    add_submodule(m_half_);
    add_weight(a_);
    add_weight(b_);
  }

  ~TestTop() override { MINFO("Destroy TestTop"); }

  void forward() override {
    MINFO("Forward TestTop");
    MDEBUG("before: a={}", a_->str());
    MDEBUG("before: b={}", b_->str());
    m_affine_->forward();
    m_half_->forward();
    MDEBUG("after: a={}", a_->str());
    MDEBUG("after: b={}", b_->str());
  }
};

// Test fixture for Module tests
class ModuleTest : public ::testing::Test {
protected:
  TensorPtr<float> tensor3a;
  TensorPtr<float> tensor3b;
  TensorPtr<float> tensor4a;
  TensorPtr<float> tensor4b;

  LoggerPtr logger_;

  void SetUp() override {
    logger_ = setup_logger("ModuleTest");
    MINFO("SetUp start");

    tensor3a = std::make_shared<Tensor<float>>("tensor3a", TensorShape({3}));
    tensor3b = std::make_shared<Tensor<float>>("tensor3b", TensorShape({3}));
    tensor4a = std::make_shared<Tensor<float>>("tensor4a", TensorShape({4}));
    tensor4b = std::make_shared<Tensor<float>>("tensor4b", TensorShape({4}));

    tensor3a->linspace(0.0f, 1.0f);
    tensor3b->linspace(2.0f, 3.0f);
    tensor4a->linspace(3.0f, 4.0f);
    tensor4b->linspace(4.0f, 5.0f);

    MINFO("SetUp end");
  }

  void TearDown() override { MINFO("TearDown"); }
};

// Test cases for Module class
TEST_F(ModuleTest, ConstructorAndBasicSetup) {
  TestAffine affine("affine", 3, tensor3a, tensor3b);
  EXPECT_EQ(affine.name(), "affine");
  EXPECT_EQ(affine.param("size"), 3);
  EXPECT_EQ(affine.inputs().size(), 1);
  EXPECT_EQ(affine.outputs().size(), 1);
  EXPECT_EQ(affine.inouts().size(), 0);
  EXPECT_EQ(affine.inputs(0)->name(), "tensor3a");
  EXPECT_EQ(affine.outputs(0)->name(), "tensor3b");
  EXPECT_EQ(affine.weights().size(), 2);
  EXPECT_EQ(affine.weights()[0]->name(), "weight1");
  EXPECT_EQ(affine.weights()[1]->name(), "bias1");
  EXPECT_EQ(affine.submodules().size(), 0);

  TestHalf half("half", 3, tensor3b);
  EXPECT_EQ(half.name(), "half");
  EXPECT_EQ(half.param("size"), 3);
  EXPECT_EQ(half.inputs().size(), 1);
  EXPECT_EQ(half.outputs().size(), 0);
  EXPECT_EQ(half.inouts().size(), 0);
  EXPECT_EQ(half.inputs(0)->name(), "tensor3b");
  EXPECT_EQ(half.weights().size(), 0);
  EXPECT_EQ(half.submodules().size(), 0);
}

TEST_F(ModuleTest, StringRepresentation) {
  TestAffine module("affine", 3, tensor3a, tensor3b);
  std::string str_rep = module.str();
  EXPECT_NE(str_rep.find("affine"), std::string::npos);
  EXPECT_NE(str_rep.find("size:3"), std::string::npos);
}

TEST_F(ModuleTest, SuperModuleConstructor) {
  TestTop top;
  EXPECT_EQ(top.name(), "top");
  EXPECT_EQ(top.inputs().size(), 0);
  EXPECT_EQ(top.outputs().size(), 0);
  EXPECT_EQ(top.inouts().size(), 0);
  EXPECT_EQ(top.submodules().size(), 2);
  EXPECT_EQ(top.submodules(0)->name(), "affine");
  EXPECT_EQ(top.submodules(1)->name(), "half");
}

TEST_F(ModuleTest, AffineForward) {
  TestAffine affine("affine", 3, tensor3a, tensor3b);
  EXPECT_EQ(tensor3b->at(0), 2.0f);
  EXPECT_EQ(tensor3b->at(1), 5.0f);
  EXPECT_EQ(tensor3b->at(2), 8.0f);
  affine.weights(0)->linspace(0.0f, 1.0f);
  affine.weights(1)->linspace(2.0f, 3.0f);
  affine.forward();
  EXPECT_EQ(tensor3b->at(0), 2.0f);
  EXPECT_EQ(tensor3b->at(1), 6.0f);
  EXPECT_EQ(tensor3b->at(2), 12.0f);
}

TEST_F(ModuleTest, HalfForward) {
  TestHalf half("half", 3, tensor3b);
  EXPECT_EQ(tensor3b->at(0), 2.0f);
  EXPECT_EQ(tensor3b->at(1), 5.0f);
  EXPECT_EQ(tensor3b->at(2), 8.0f);
  half.forward();
  EXPECT_EQ(tensor3b->at(0), 1.0f);
  EXPECT_EQ(tensor3b->at(1), 2.5f);
  EXPECT_EQ(tensor3b->at(2), 4.0f);
}

TEST_F(ModuleTest, TopForward) {
  TestTop top;
  top.weights(0)->linspace(0.0f, 1.0f);
  top.weights(1)->linspace(2.0f, 3.0f);
  top.submodules(0)->weights(0)->linspace(0.0f, 1.0f);
  top.submodules(0)->weights(1)->linspace(2.0f, 3.0f);
  EXPECT_EQ(top.weights(0)->at(0), 0.0f);
  EXPECT_EQ(top.weights(0)->at(1), 1.0f);
  EXPECT_EQ(top.weights(0)->at(2), 2.0f);
  EXPECT_EQ(top.weights(1)->at(0), 2.0f);
  EXPECT_EQ(top.weights(1)->at(1), 5.0f);
  EXPECT_EQ(top.weights(1)->at(2), 8.0f);
  top.forward();
  EXPECT_EQ(top.weights(0)->at(0), 0.0f);
  EXPECT_EQ(top.weights(0)->at(1), 1.0f);
  EXPECT_EQ(top.weights(0)->at(2), 2.0f);
  EXPECT_EQ(top.weights(1)->at(0), 1.0f);
  EXPECT_EQ(top.weights(1)->at(1), 3.0f);
  EXPECT_EQ(top.weights(1)->at(2), 6.0f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
