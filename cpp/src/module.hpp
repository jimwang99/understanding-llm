#pragma once

#include "tensor.hpp"
#include <map>
#include <string>

#include "logger.hpp"

class Module; // forward declarations
using ModulePtr = std::shared_ptr<Module>;

class Module {
protected:
  std::string name_;
  std::map<std::string, size_t> params_;     // hyper parameters
  std::vector<const TensorBasePtr> weights_; // trainable weights
  std::vector<const TensorBasePtr> hiddens_; // hidden states' tensors
  std::vector<const CTensorBasePtr> inputs_; // input tensors
  std::vector<const TensorBasePtr> outputs_; // output tensors
  std::vector<const TensorBasePtr> inouts_;  // input/output tensors
  std::vector<const ModulePtr> submodules_;  // sub-modules

  LoggerPtr logger_;

  //===========================================================================
  // Modifiers
  //===========================================================================
  void set_name(const std::string name) { name_ = name; }
  void add_param(const std::string name, const size_t value) {
    params_[name] = value;
  }
  void add_weight(const TensorBasePtr weight) { weights_.push_back(weight); }
  void add_hidden(const TensorBasePtr hidden) { hiddens_.push_back(hidden); }
  void add_input(const CTensorBasePtr input) { inputs_.push_back(input); }
  void add_output(const TensorBasePtr output) { outputs_.push_back(output); }
  void add_inout(const TensorBasePtr inout) { inouts_.push_back(inout); }
  void add_submodule(const ModulePtr submodule) {
    submodules_.push_back(submodule);
  }

public:
  //===========================================================================
  // Constructors
  //===========================================================================
  // default constructor
  Module() : Module("M") {}
  Module(const std::string name)
      : name_(name), params_(), weights_(), hiddens_(), inputs_(), outputs_(),
        inouts_(), submodules_() {
    logger_ = get_logger();
  }
  virtual ~Module() = default;

  //===========================================================================
  // Accessors
  //===========================================================================
  const std::string &name() const { return name_; }
  size_t param(const std::string name) const { return params_.at(name); }
  const TensorBasePtr &weights(const size_t i) const { return weights_.at(i); }
  const TensorBasePtr &hiddens(const size_t i) const { return hiddens_.at(i); }
  const CTensorBasePtr &inputs(const size_t i) const { return inputs_.at(i); }
  const TensorBasePtr &outputs(const size_t i) const { return outputs_.at(i); }
  const TensorBasePtr &inouts(const size_t i) const { return inouts_.at(i); }
  const ModulePtr &submodules(const size_t i) const {
    return submodules_.at(i);
  }
  const std::vector<const TensorBasePtr> &weights() const { return weights_; }
  const std::vector<const TensorBasePtr> &hiddens() const { return hiddens_; }
  const std::vector<const CTensorBasePtr> &inputs() const { return inputs_; }
  const std::vector<const TensorBasePtr> &outputs() const { return outputs_; }
  const std::vector<const TensorBasePtr> &inouts() const { return inouts_; }
  const std::vector<const ModulePtr> &submodules() const { return submodules_; }

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
    std::string s = fmt::format("Module name={}", name_);
    if (!params_.empty()) {
      s += " param=[";
      for (auto &[name, param] : params_) {
        s += fmt::format("{}:{},", name, param);
      }
      s.back() = ']';
    }
    for (auto &weight : weights_) {
      s += fmt::format("\n  weight {}", weight->str());
    }
    for (auto &hidden : hiddens_) {
      s += fmt::format("\n  hidden {}", hidden->str());
    }
    for (auto &input : inputs_) {
      s += fmt::format("\n  input  {}", input->str());
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
}; // class Module
