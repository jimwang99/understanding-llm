#pragma once

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <spdlog/spdlog.h>

#include <cassert>

#define ASSERT(condition, ...) \
  if (!(condition)) {          \
    CRITICAL(__VA_ARGS__);     \
    assert(condition);         \
  }

std::shared_ptr<spdlog::logger> setup_logger(
    const std::string name = "default",
    const std::string level = SPDLOG_ACTIVE_LEVEL,
    const std::vector<spdlog::sink_ptr> sinks = {});

std::shared_ptr<spdlog::logger> get_logger(const std::string name = "default");

#define CRITICAL(...) SPDLOG_LOGGER_CRITICAL(this->logger_, __VA_ARGS__)
#define ERROR(...) SPDLOG_LOGGER_ERROR(this->logger_, __VA_ARGS__)
#define WARN(...) SPDLOG_LOGGER_WARN(this->logger_, __VA_ARGS__)
#define INFO(...) SPDLOG_LOGGER_INFO(this->logger_, __VA_ARGS__)
#define DEBUG(...) SPDLOG_LOGGER_DEBUG(this->logger_, __VA_ARGS__)
#define TRACE(...) SPDLOG_LOGGER_TRACE(this->logger_, __VA_ARGS__)