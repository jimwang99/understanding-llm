#include "../src/logger.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

class LoggerTest : public ::testing::Test {
protected:
  std::shared_ptr<std::ostringstream> log_stream;
  std::shared_ptr<spdlog::sinks::ostream_sink_mt> stream_sink;
  std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> stdout_sink;
  std::shared_ptr<spdlog::sinks::basic_file_sink_mt> file_sink;

  void SetUp() override {
    // Clear any existing loggers before each test
    spdlog::drop_all();
    // Create a stringstream sink for capturing log messages
    log_stream = std::make_shared<std::ostringstream>();
    stream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(*log_stream);
    // Create a stdout color sink for logging to the console
    stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // Create a file sink for logging to a file
    file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "/tmp/test_logger.log", true);
  }

  void TearDown() override {
    // Clean up
    spdlog::drop_all();
  }

  // Helper to check if a string contains a substring
  bool contains(const std::string &haystack, const std::string &needle) {
    return haystack.find(needle) != std::string::npos;
  }
};

// Test logger setup with default parameters
TEST_F(LoggerTest, DefaultSetupLogger) {
  auto logger = get_logger();
  ASSERT_NE(logger, nullptr);
  EXPECT_EQ(logger->name(), "default");
  EXPECT_EQ(logger->level(), spdlog::level::from_str(LOG_LEVEL));
}

// Test logger setup with custom name and level
TEST_F(LoggerTest, CustomSetupLogger) {
  auto logger = get_logger("custom_logger", "error");
  ASSERT_NE(logger, nullptr);
  EXPECT_EQ(logger->name(), "custom_logger");
  EXPECT_EQ(logger->level(), spdlog::level::err);
}

// Test logger setup with custom sink
TEST_F(LoggerTest, SetupLoggerWithSink) {
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  auto logger = get_logger("sink_logger", "info", sinks);

  ASSERT_NE(logger, nullptr);
  EXPECT_EQ(logger->name(), "sink_logger");
  EXPECT_EQ(logger->level(), spdlog::level::info);

  // Log a test message
  logger->info("test message with level at info");
  logger->debug("test message with level at debug");

  // Verify the message was logged to our stream
  std::string log_content = log_stream->str();
  EXPECT_TRUE(contains(log_content, "test message with level at info"));
  EXPECT_FALSE(contains(log_content, "test message with level at debug"));
}

// Test get_logger retrieves the correct logger
TEST_F(LoggerTest, GetLogger) {
  auto logger1 = get_logger("test_get", "debug");
  auto logger2 = get_logger("test_get");

  ASSERT_NE(logger2, nullptr);
  EXPECT_EQ(logger2->name(), "test_get");
  EXPECT_EQ(logger2->level(), spdlog::level::debug);

  // Should be the same logger instance
  EXPECT_EQ(logger1, logger2);
}

// Test for the logging macros
class LoggableClass {
public:
  LoggableClass(const std::vector<spdlog::sink_ptr> &sinks) {
    logger_ = get_logger("class_logger", "trace", sinks);
  }

  void testLogs() {
    MLOG_CRITICAL("Critical message");
    MLOG_ERROR("Error message");
    MLOG_WARN("Warning message");
    MLOG_INFO("Info message");
    MLOG_DEBUG("Debug message");
    MLOG_TRACE("Trace message");
  }

  void testAssert(bool condition) { MASSERT(condition, "Assert failed"); }

private:
  std::shared_ptr<spdlog::logger> logger_;
};

TEST_F(LoggerTest, MacroLogging) {
  // Set up a test logger with our stream sink
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  auto logger = get_logger("default", "trace", sinks);
  EXPECT_EQ(logger->name(), "default");
  EXPECT_EQ(logger->level(), spdlog::level::trace);

  // Log messages using the global macros
  LOG_CRITICAL("Global critical");
  LOG_ERROR("Global error");
  LOG_WARN("Global warn");
  LOG_INFO("Global info");
  LOG_DEBUG("Global debug");
  LOG_TRACE("Global trace");

  // Verify at least some logs were generated
  std::string log_content = log_stream->str();
  EXPECT_FALSE(log_content.empty());
  EXPECT_TRUE(contains(log_content, "Global critical"));
  EXPECT_TRUE(contains(log_content, "Global error"));
  EXPECT_TRUE(contains(log_content, "Global warn"));
  EXPECT_TRUE(contains(log_content, "Global info"));
  EXPECT_TRUE(contains(log_content, "Global debug"));
  EXPECT_TRUE(contains(log_content, "Global trace"));
}

TEST_F(LoggerTest, MacroLoggingWithArgs) {
  // Set up a test logger with our stream sink
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  auto logger = get_logger("default", "trace", sinks);

  // Log messages using the global macros with arguments
  LOG_CRITICAL("Global critical with arg: {}", 42);
  LOG_ERROR("Global error with arg: {}", 42);
  LOG_WARN("Global warn with arg: {}", 42);
  LOG_INFO("Global info with arg: {}", 42);
  LOG_DEBUG("Global debug with arg: {}", 42);
  LOG_TRACE("Global trace with arg: {}", 42);

  // Verify logs were generated
  std::string log_content = log_stream->str();
  EXPECT_FALSE(log_content.empty());
  EXPECT_TRUE(contains(log_content, "Global critical with arg: 42"));
  EXPECT_TRUE(contains(log_content, "Global error with arg: 42"));
  EXPECT_TRUE(contains(log_content, "Global warn with arg: 42"));
  EXPECT_TRUE(contains(log_content, "Global info with arg: 42"));
}

TEST_F(LoggerTest, MacroLoggingWithName) {
  // Set up a test logger with our stream sink
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  auto logger_name = "custom_logger";
  auto logger = get_logger(logger_name, "trace", sinks);
  EXPECT_EQ(logger->name(), logger_name);
  EXPECT_EQ(logger->level(), spdlog::level::trace);

  // Log messages using the global macros
  NLOG_CRITICAL("Global critical");
  NLOG_ERROR("Global error");
  NLOG_WARN("Global warn");
  NLOG_INFO("Global info");
  NLOG_DEBUG("Global debug");
  NLOG_TRACE("Global trace");

  // Verify at least some logs were generated
  std::string log_content = log_stream->str();
  EXPECT_FALSE(log_content.empty());
  EXPECT_TRUE(contains(log_content, "Global critical"));
  EXPECT_TRUE(contains(log_content, "Global error"));
  EXPECT_TRUE(contains(log_content, "Global warn"));
  EXPECT_TRUE(contains(log_content, "Global info"));
  EXPECT_TRUE(contains(log_content, "Global debug"));
  EXPECT_TRUE(contains(log_content, "Global trace"));
}

// Test for ASSERT macro
TEST_F(LoggerTest, AssertMacro) {
#ifdef NDEBUG
  std::cout << "NDEBUG is defined - assertions are disabled\n";
#else
  std::cout << "NDEBUG is not defined - assertions are enabled\n";
#endif
  // Set up a test logger with our stream sink
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  auto logger = get_logger("default", "trace", sinks);

  // Call the assert method with true (should not log or assert)
  ASSERT(true, "Assert failed");
  std::string log_content = log_stream->str();
  EXPECT_TRUE(log_content.empty());

  EXPECT_DEATH(ASSERT(false, "Assert failed"),
               "Assertion failed.*test_logger.cpp.*");
}

TEST_F(LoggerTest, ClassMacroLogging) {
  // Create an instance of our test class
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  LoggableClass obj(sinks);

  // Call the method that logs
  obj.testLogs();

  // Verify logs were generated
  std::string log_content = log_stream->str();
  EXPECT_FALSE(log_content.empty());
  EXPECT_TRUE(contains(log_content, "Critical message"));
  EXPECT_TRUE(contains(log_content, "Error message"));
  EXPECT_TRUE(contains(log_content, "Warning message"));
  EXPECT_TRUE(contains(log_content, "Info message"));
  EXPECT_TRUE(contains(log_content, "Debug message"));
  EXPECT_TRUE(contains(log_content, "Trace message"));
}

// Test for MASSERT macro
TEST_F(LoggerTest, MassertMacro) {
  // Create an instance of our test class
  std::vector<spdlog::sink_ptr> sinks{stream_sink, stdout_sink, file_sink};
  LoggableClass obj(sinks);

  // Call the assert method with true (should not log or assert)
  obj.testAssert(true);

  // Verify logs were generated
  std::string log_content = log_stream->str();
  EXPECT_TRUE(log_content.empty());
  EXPECT_DEATH(obj.testAssert(false), "Assertion failed.*test_logger.cpp.*");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
