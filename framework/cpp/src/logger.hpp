#pragma once

#include <spdlog/spdlog.h>

#include <cassert>

#ifndef LOG_LEVEL_DEFAULT
#define LOG_LEVEL_DEFAULT "trace"
#endif

#define ASSERT(condition, ...) \
  if (!(condition)) {          \
    CRITICAL(__VA_ARGS__);     \
    assert(condition);         \
  }

std::shared_ptr<spdlog::logger> setup_logger(
    const std::string name = "default",
    const std::string level = LOG_LEVEL_DEFAULT,
    const std::vector<spdlog::sink_ptr> sinks = {});

std::shared_ptr<spdlog::logger> get_logger(const std::string name = "default");

#define CRITICAL(...) SPDLOG_LOGGER_CRITICAL(this->logger_, __VA_ARGS__)
#define ERROR(...) SPDLOG_LOGGER_ERROR(this->logger_, __VA_ARGS__)
#define WARN(...) SPDLOG_LOGGER_WARN(this->logger_, __VA_ARGS__)
#define INFO(...) SPDLOG_LOGGER_INFO(this->logger_, __VA_ARGS__)
#define DEBUG(...) SPDLOG_LOGGER_DEBUG(this->logger_, __VA_ARGS__)
#define TRACE(...) SPDLOG_LOGGER_TRACE(this->logger_, __VA_ARGS__)