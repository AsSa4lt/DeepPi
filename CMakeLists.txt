cmake_minimum_required(VERSION 3.12)
project(DeepPiTests)

# Set the C++ standard.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable testing.
enable_testing()

# Find GoogleTest.
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Create the test executables for each test file.
add_executable(test_tensors tests/tensorTests/test_access.cpp 
                            tests/tensorTests/test_sum.cpp 
                            tests/tensorTests/test_substraction.cpp
                            tests/tensorTests/test_tensorops.cpp)

# Link with GoogleTest and pthread for each test executable.
target_link_libraries(test_tensors GTest::GTest GTest::Main pthread)

# Register the tests with CTest.
add_test(NAME test_tensors COMMAND test_tensors)