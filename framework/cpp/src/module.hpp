#pragma once

#include "tensor.hpp"
// #include <fstream>
#include <map>
#include <string>

#include "logger.hpp"

template <typename T>
class Module {
 protected:
  std::string name_;
  std::map<std::string, size_t> params_;             // hyper parameters
  std::vector<std::shared_ptr<Tensor<T>>> weights_;  // trainable weights
  std::vector<std::shared_ptr<Tensor<T>>> inputs_;   // input tensors
  std::vector<std::shared_ptr<Tensor<T>>> outputs_;  // output tensors
  std::vector<std::shared_ptr<Tensor<T>>> inouts_;   // input/output tensors
  std::vector<std::shared_ptr<Module<T>>> submodules_;

  std::shared_ptr<spdlog::logger> logger_;

 public:
  //===========================================================================
  // Constructors
  //===========================================================================
  // default constructor
  Module(const std::string name = "Unnamed-Module")
      : name_(name),
        params_(),
        weights_(),
        inputs_(),
        outputs_(),
        inouts_(),
        submodules_() {
    logger_ = setup_logger();
  }

  //===========================================================================
  // Modifiers
  //===========================================================================
  void set_name(const std::string name) { name_ = name; }
  void add_param(const std::string name, const size_t value) {
    params_[name] = value;
  }
  void add_weight(Tensor<T> &weight) {
    weights_.push_back(std::make_shared<Tensor<T>>(weight));
  }
  void add_input(Tensor<T> &input) {
    inputs_.push_back(std::make_shared<Tensor<T>>(input));
  }
  void add_output(Tensor<T> &output) {
    outputs_.push_back(std::make_shared<Tensor<T>>(output));
  }
  void add_inout(Tensor<T> &inout) {
    inouts_.push_back(std::make_shared<Tensor<T>>(inout));
  }
  void add_submodule(Module<T> &submodule) {
    submodules_.push_back(std::make_shared<Module<T>>(submodule));
  }

  //----------------------------------------------------------------------------
  // Weights
  //----------------------------------------------------------------------------
  // // load one weight tensor from file
  // void load_weight_from_file(const std::string &name, const std::string
  // &path) {
  //   // open file in binary mode, and read the file contents into the weights_
  //   spdlog::info("[Module {}] Loading weight {} from {}", name_, name, path);
  //   std::ifstream f(path, std::ios::binary);
  //   auto nbytes = f.tellg();
  //   assert(nbytes == weights_[name].nbytes());
  //   f.seekg(0, std::ios::beg);
  //   f.read(reinterpret_cast<char *>(weights_[name].data()), nbytes);
  //   f.close();
  // }
  // // load one weight tensor from a buffer
  // void load_weight_from_buffer(const std::string &name, std::vector<T>
  // buffer,
  //                  size_t offset) {
  //   auto w = weights_[name];
  //   memcpy(w.data(), buffer.data() + offset, w.nbytes());
  // }
  // // dump one weight tensor to file
  // void dump_weight_to_file(const std::string &name, const std::string &path)
  // {
  //   spdlog::info("[Module {}] Dumping weight {} to {}", name_, name, path);
  //   // open file in binary mode, and write the file contents from the
  //   weights_ std::ofstream f(path, std::ios::binary);
  //   f.write(reinterpret_cast<const char *>(weights_[name].data()),
  //           weights_[name].nbytes());
  // }
  // // load all weight tensors from files in a directory
  // void load_weights(const std::string &path) {
  //   // TODO: implement
  // }
  // // dump all weight tensors to files in a directory
  // void dump_weights(const std::string &path) {
  //   // TODO: implement
  // }

  //----------------------------------------------------------------------------
  // Submodules
  //----------------------------------------------------------------------------
  void add_submodule(Module<T> &submodule) {
    auto name = submodule.get_name();
    assert(submodules_.find(name) == submodules_.end());  // check name unique
    submodules_[name] = submodule;
    submodule.set_name(name_ + "." + name);
  }

  //----------------------------------------------------------------------------
  // inputs and outputs
  //----------------------------------------------------------------------------
  void add_input(const Tensor<T> &input) {
    inputs_.push_back(std::make_unique<Tensor<T>>(input));
  }
  void add_output(const Tensor<T> &output) {
    outputs_.push_back(std::make_unique<Tensor<T>>(output));
  }

  //===========================================================================
  // Accessors
  //===========================================================================
  const std::string get_name() { return name_; }
  Module<T> &get_submodule(const std::string name) { return submodules_[name]; }

  //===========================================================================
  // Pack and unpack
  //===========================================================================
  void pack_weights() {
    // TODO: implement
  }
  void unpack_weights() {
    // TODO: implement
  }
  void pack_inputs() {
    // TODO: implement
  }
  void unpack_inputs() {
    // TODO: implement
  }
  void pack_outputs() {
    // TODO: implement
  }
  void unpack_outputs() {
    // TODO: implement
  }

  //===========================================================================
  // Pretty print
  //===========================================================================
  std::string str() const {
    std::string s = fmt::format("Module name={} param=[", name_);
    for (auto &[name, param] : params_) {
      s += fmt::format("{}:{},", name, param);
    }
    s.back() = ']';
    for (auto &weight : weights_) {
      s += fmt::format("\n  weight {}", weight->str());
    }
    for (auto &input : inputs_) {
      s += fmt::format("\n  input {}", input->str());
    }
    for (auto &output : outputs_) {
      s += fmt::format("\n  output {}", output->str());
    }
    for (auto &submodule : submodules_) {
      s += fmt::format("\n{}", submodule->str());
    }
    return s;
  }

  //===========================================================================
  // Inference
  //===========================================================================
  virtual void forward() = 0;
};  // class Module
