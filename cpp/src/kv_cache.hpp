namespace module {

template <typename T> class KVCache : public virtual Module<T> {
private:
  const size_t max_seq_len_;
  const size_ head_dim_;

  Tensor<T> &cache_;
  Tensor<T> &kv_;

public:
  KVCache(const std::string name, const size_t max_seq_len, const head_dim,
          Tensor<T> &cache, Tensor<T> &kv)
      : Module<T>(name), max_seq_len_(max_seq_len), head_dim_(head_dim),
        cache_(cache), kv_(kv), logger_(get_logger("KVCache")) {
    add_param("max_seq_len", max_seq_len_);
    add_param("head_dim", head_dim_);
    add_inout(cache_);
    add_inout(kv_);
    DEBUG("New module of KVCache:\n{}", this->str());
  }

  void forward() {
    // TODO
  }
};

} // namespace module