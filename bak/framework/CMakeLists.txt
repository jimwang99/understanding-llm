cmake_minimum_required(VERSION 3.12)
project(example)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)

# Create the Python module
pybind11_add_module(test_tensor_lib test_tensor_lib.cpp)
target_link_libraries(test_tensor_lib PRIVATE spdlog::spdlog)
target_compile_options(test_tensor_lib PRIVATE -O0)
