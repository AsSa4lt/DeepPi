cmake_minimum_required(VERSION 3.12)
project(DeepPiTests)

# Set the C++ standard.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS ON) 

# Enable testing.
enable_testing()

# Find GoogleTest.
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)



# Create the test executables for each test file.
add_executable(test_tensors tests/tensorTests/test_access.cpp 
                            tests/tensorTests/test_sum.cpp 
                            tests/tensorTests/test_substraction.cpp
                            tests/tensorTests/test_tensorops.cpp
                            tests/tensorTests/test_matmul.cpp)

# Add sources
target_sources(test_tensors PUBLIC src/TensorMatmul.cpp)

# Add compile options
target_compile_options(test_tensors PUBLIC -O3)

# Link with GoogleTest and pthread for each test executable.
target_link_libraries(test_tensors GTest::GTest GTest::Main pthread)

# Register the tests with CTest.
add_test(NAME test_tensors COMMAND test_tensors)