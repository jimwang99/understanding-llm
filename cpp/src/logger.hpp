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

LoggerPtr setup_logger(const std::string &name = "default",
                       const std::string &level = LOG_LEVEL,
                       const std::vector<spdlog::sink_ptr> &sinks = {});

LoggerPtr get_logger(const std::string &name = "default");

#define MCRITICAL(...) SPDLOG_LOGGER_CRITICAL(this->logger_, __VA_ARGS__)
#define MERROR(...) SPDLOG_LOGGER_ERROR(this->logger_, __VA_ARGS__)
#define MWARN(...) SPDLOG_LOGGER_WARN(this->logger_, __VA_ARGS__)
#define MINFO(...) SPDLOG_LOGGER_INFO(this->logger_, __VA_ARGS__)
#define MDEBUG(...) SPDLOG_LOGGER_DEBUG(this->logger_, __VA_ARGS__)
#define MTRACE(...) SPDLOG_LOGGER_TRACE(this->logger_, __VA_ARGS__)

#define CRITICAL(...) SPDLOG_LOGGER_CRITICAL(get_logger(), __VA_ARGS__)
#define ERROR(...) SPDLOG_LOGGER_ERROR(get_logger(), __VA_ARGS__)
#define WARN(...) SPDLOG_LOGGER_WARN(get_logger(), __VA_ARGS__)
#define INFO(...) SPDLOG_LOGGER_INFO(get_logger(), __VA_ARGS__)
#define DEBUG(...) SPDLOG_LOGGER_DEBUG(get_logger(), __VA_ARGS__)
#define TRACE(...) SPDLOG_LOGGER_TRACE(get_logger(), __VA_ARGS__)

#define MASSERT(condition, ...)                                                \
  if (!(condition)) {                                                          \
    MCRITICAL(__VA_ARGS__);                                                    \
    assert(condition);                                                         \
  }

#define ASSERT(condition, ...)                                                 \
  if (!(condition)) {                                                          \
    CRITICAL(__VA_ARGS__);                                                     \
    assert(condition);                                                         \
  }
