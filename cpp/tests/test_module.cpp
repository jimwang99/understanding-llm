#include "logger.hpp"
#include "module.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>

class TestAffine final : public Module {
public:
  CTensorPtr<float> in_;
  TensorPtr<float> out_;
  TensorPtr<float> weight_;
  TensorPtr<float> bias_;

  TestAffine(const std::string name, const size_t size, CTensorPtr<float> in,
             TensorPtr<float> out)
      : Module(name), in_(in), out_(out),
        weight_(std::make_shared<Tensor<float>>("weight", TensorShape({size}))),
        bias_(std::make_shared<Tensor<float>>("bias", TensorShape({size}))) {
    MLOG_DEBUG("Create TestAffine");
    add_param("size", size);
    add_weight(weight_);
    add_weight(bias_);
    add_input(in_);
    add_output(out_);
    MASSERT(in_->size() == size, "Input size {} must match size {}",
            in_->size(), size);
    MASSERT(out_->size() == size, "Output size {} must match size {}",
            out_->size(), size);
    MLOG_DEBUG("TestAffine is created: \n{}", str());
  }

  ~TestAffine() override { MLOG_DEBUG("Destroy TestAffine"); }

  void forward() override {
    MLOG_DEBUG("Forward TestAffine");
    MLOG_DEBUG("before: in={}", in_->str());
    MLOG_DEBUG("before: weight={}", weight_->str());
    MLOG_DEBUG("before: bias={}", bias_->str());
    MLOG_DEBUG("before: out={}", out_->str());
    for (size_t i = 0; i < param("size"); ++i) {
      out_->at(i) = in_->at(i) * weight_->at(i) + bias_->at(i);
    }
    MLOG_DEBUG("after: in={}", in_->str());
    MLOG_DEBUG("after: weight={}", weight_->str());
    MLOG_DEBUG("after: bias={}", bias_->str());
    MLOG_DEBUG("after: out={}", out_->str());
  }
};

class TestHalf final : public Module {
public:
  TensorPtr<float> inout_;

  TestHalf(const std::string name, const size_t size, TensorPtr<float> inout)
      : Module(name), inout_(inout) {
    MLOG_DEBUG("Create TestHalf");
    add_param("size", size);
    add_input(inout_);
    MASSERT(
        inout_->size() == size,
        fmt::format("Input size {} must match size {}", inout_->size(), size));
    MLOG_DEBUG("TestHalf is created: \n{}", str());
  }

  ~TestHalf() override { MLOG_DEBUG("Destroy TestHalf"); }

  void forward() override {
    MLOG_DEBUG("Forward TestHalf");
    MLOG_DEBUG("before: inout={}", inout_->str());
    for (size_t i = 0; i < inout_->size(); ++i) {
      inout_->at(i) = inout_->at(i) / 2.0f;
    }
    MLOG_DEBUG("after: inout={}", inout_->str());
  }
};

class TestTop final : public Module {
public:
  TensorPtr<float> a_;
  TensorPtr<float> b_;
  std::shared_ptr<TestAffine> m_affine_;
  std::shared_ptr<TestHalf> m_half_;

  TestTop()
      : Module("top"),
        a_(std::make_shared<Tensor<float>>("a", TensorShape({3}))),
        b_(std::make_shared<Tensor<float>>("b", TensorShape({3}))),
        m_affine_(std::make_shared<TestAffine>("affine", 3, a_, b_)),
        m_half_(std::make_shared<TestHalf>("half", 3, b_)) {
    MLOG_DEBUG("Create TestTop");
    add_hidden(a_);
    add_hidden(b_);
    add_submodule(std::static_pointer_cast<Module>(m_affine_));
    add_submodule(std::static_pointer_cast<Module>(m_half_));
    MLOG_DEBUG("TestTop is created: {}", str());
  }

  ~TestTop() override { MLOG_DEBUG("Destroy TestTop"); }

  void forward() override {
    MLOG_DEBUG("Forward TestTop");
    MLOG_DEBUG("before: a={}", a_->str());
    MLOG_DEBUG("before: b={}", b_->str());
    m_affine_->forward();
    m_half_->forward();
    MLOG_DEBUG("after: a={}", a_->str());
    MLOG_DEBUG("after: b={}", b_->str());
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
    logger_ = get_logger("ModuleTest");
    MLOG_DEBUG("SetUp start");

    tensor3a = std::make_shared<Tensor<float>>("tensor3a", TensorShape({3}));
    tensor3b = std::make_shared<Tensor<float>>("tensor3b", TensorShape({3}));
    tensor4a = std::make_shared<Tensor<float>>("tensor4a", TensorShape({4}));
    tensor4b = std::make_shared<Tensor<float>>("tensor4b", TensorShape({4}));

    tensor3a->linspace(0.0f, 1.0f);
    tensor3b->linspace(2.0f, 3.0f);
    tensor4a->linspace(3.0f, 4.0f);
    tensor4b->linspace(4.0f, 5.0f);

    MLOG_DEBUG("SetUp end");
  }

  void TearDown() override { MLOG_DEBUG("TearDown"); }
};

// Test cases for Module class
TEST_F(ModuleTest, ConstructorAndBasicSetup) {
  TestAffine affine("affine", 3, tensor3a, tensor3b);
  EXPECT_EQ(affine.name(), "affine");
  EXPECT_EQ(affine.param("size"), 3u);
  EXPECT_EQ(affine.inputs().size(), 1u);
  EXPECT_EQ(affine.outputs().size(), 1u);
  EXPECT_EQ(affine.inouts().size(), 0u);
  EXPECT_EQ(affine.inputs(0)->name(), "tensor3a");
  EXPECT_EQ(affine.outputs(0)->name(), "tensor3b");
  EXPECT_EQ(affine.weights().size(), 2u);
  EXPECT_EQ(affine.weights()[0]->name(), "weight");
  EXPECT_EQ(affine.weights()[1]->name(), "bias");
  EXPECT_EQ(affine.submodules().size(), 0u);

  TestHalf half("half", 3, tensor3b);
  EXPECT_EQ(half.name(), "half");
  EXPECT_EQ(half.param("size"), 3u);
  EXPECT_EQ(half.inputs().size(), 1u);
  EXPECT_EQ(half.outputs().size(), 0u);
  EXPECT_EQ(half.inouts().size(), 0u);
  EXPECT_EQ(half.inputs(0)->name(), "tensor3b");
  EXPECT_EQ(half.weights().size(), 0u);
  EXPECT_EQ(half.submodules().size(), 0u);
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
  EXPECT_EQ(top.inputs().size(), 0u);
  EXPECT_EQ(top.outputs().size(), 0u);
  EXPECT_EQ(top.inouts().size(), 0u);
  EXPECT_EQ(top.hiddens().size(), 2u);
  EXPECT_EQ(top.submodules().size(), 2u);
  EXPECT_EQ(top.hiddens(0)->name(), "a");
  EXPECT_EQ(top.hiddens(1)->name(), "b");
  EXPECT_EQ(top.submodules(0)->name(), "affine");
  EXPECT_EQ(top.submodules(1)->name(), "half");
  EXPECT_EQ(top.a_.use_count(), 4u);
  EXPECT_EQ(top.b_.use_count(), 6u);
  EXPECT_EQ(top.m_affine_.use_count(), 2u);
  EXPECT_EQ(top.m_half_.use_count(), 2u);
}

TEST_F(ModuleTest, AffineForward) {
  TestAffine affine("affine", 3, tensor3a, tensor3b);
  EXPECT_EQ(tensor3b->at(0), 2.0f);
  EXPECT_EQ(tensor3b->at(1), 5.0f);
  EXPECT_EQ(tensor3b->at(2), 8.0f);
  affine.weight_->linspace(0.0f, 1.0f);
  affine.bias_->linspace(2.0f, 3.0f);
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
  top.a_->linspace(0.0f, 1.0f);
  top.b_->linspace(2.0f, 3.0f);
  top.m_affine_->weight_->linspace(0.0f, 1.0f);
  top.m_affine_->bias_->linspace(2.0f, 3.0f);
  EXPECT_EQ(top.m_affine_->weight_->at(0), 0.0f);
  EXPECT_EQ(top.m_affine_->weight_->at(1), 1.0f);
  top.forward();
  EXPECT_EQ(top.a_->at(0), 0.0f);
  EXPECT_EQ(top.a_->at(1), 1.0f);
  EXPECT_EQ(top.a_->at(2), 2.0f);
  EXPECT_EQ(top.b_->at(0), 1.0f);
  EXPECT_EQ(top.b_->at(1), 3.0f);
  EXPECT_EQ(top.b_->at(2), 6.0f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
