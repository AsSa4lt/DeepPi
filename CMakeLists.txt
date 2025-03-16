cmake_minimum_required(VERSION 3.12)
project(DeepPiTests)

# Set the C++ standard.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing.
enable_testing()

# Find GoogleTest.
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Create the test executables for each test file.
add_executable(test_access tests/tensorTests/test_access.cpp)
add_executable(test_sum tests/tensorTests/test_sum.cpp)

# Link with GoogleTest and pthread for each test executable.
target_link_libraries(test_access GTest::GTest GTest::Main pthread)
target_link_libraries(test_sum GTest::GTest GTest::Main pthread)

# Register the tests with CTest.
add_test(NAME test_access COMMAND test_access)
add_test(NAME test_contraction COMMAND test_contraction)
