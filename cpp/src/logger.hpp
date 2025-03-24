#pragma once

#ifndef LOG_LEVEL
#define LOG_LEVEL "trace"
#endif

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <cassert>
#include <spdlog/spdlog.h>

typedef std::shared_ptr<spdlog::logger> LoggerPtr;

LoggerPtr get_logger(const std::string &name = "default",
                     const std::string &level = LOG_LEVEL,
                     const std::vector<spdlog::sink_ptr> &sinks = {});

#define LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(get_logger(), __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_ERROR(get_logger(), __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_WARN(get_logger(), __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_INFO(get_logger(), __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(get_logger(), __VA_ARGS__)
#define LOG_TRACE(...) SPDLOG_LOGGER_TRACE(get_logger(), __VA_ARGS__)

#define CRICITAL LOG_CRITICAL
#define ERROR LOG_ERROR
#define WARN LOG_WARN
#define INFO LOG_INFO
#define DEBUG LOG_DEBUG
#define TRACE LOG_TRACE

// Macro for logging within a class
#define MLOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(this->logger_, __VA_ARGS__)
#define MLOG_ERROR(...) SPDLOG_LOGGER_ERROR(this->logger_, __VA_ARGS__)
#define MLOG_WARN(...) SPDLOG_LOGGER_WARN(this->logger_, __VA_ARGS__)
#define MLOG_INFO(...) SPDLOG_LOGGER_INFO(this->logger_, __VA_ARGS__)
#define MLOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(this->logger_, __VA_ARGS__)
#define MLOG_TRACE(...) SPDLOG_LOGGER_TRACE(this->logger_, __VA_ARGS__)

// Macro for logging with a specific logger name, usually for local logging
#define NLOG_CRITICAL(...)                                                     \
  SPDLOG_LOGGER_CRITICAL(get_logger(logger_name), __VA_ARGS__)
#define NLOG_ERROR(...)                                                        \
  SPDLOG_LOGGER_ERROR(get_logger(logger_name), __VA_ARGS__)
#define NLOG_WARN(...) SPDLOG_LOGGER_WARN(get_logger(logger_name), __VA_ARGS__)
#define NLOG_INFO(...) SPDLOG_LOGGER_INFO(get_logger(logger_name), __VA_ARGS__)
#define NLOG_DEBUG(...)                                                        \
  SPDLOG_LOGGER_DEBUG(get_logger(logger_name), __VA_ARGS__)
#define NLOG_TRACE(...)                                                        \
  SPDLOG_LOGGER_TRACE(get_logger(logger_name), __VA_ARGS__)

#define ASSERT(condition, ...)                                                 \
  if (!(condition)) {                                                          \
    LOG_CRITICAL(__VA_ARGS__);                                                 \
    assert(condition);                                                         \
  }

#define MASSERT(condition, ...)                                                \
  if (!(condition)) {                                                          \
    MLOG_CRITICAL(__VA_ARGS__);                                                \
    assert(condition);                                                         \
  }

#define NASSERT(condition, ...)                                                \
  if (!(condition)) {                                                          \
    NLOG_CRITICAL(__VA_ARGS__);                                                \
    assert(condition);                                                         \
  }
