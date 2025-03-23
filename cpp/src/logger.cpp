#include "logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

LoggerPtr setup_logger(const std::string &name, const std::string &level,
                       const std::vector<spdlog::sink_ptr> &sinks) {
  auto logger = spdlog::get(name);

  if (logger == nullptr) {
    if (sinks.empty()) {
      logger = spdlog::stdout_color_mt(name);
    } else {
      logger =
          std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
      spdlog::register_logger(logger);
    }
  }
  logger->set_level(spdlog::level::from_str(level));
  return logger;
}

LoggerPtr get_logger(const std::string &name) { return spdlog::get(name); }
