cmake_minimum_required(VERSION 3.12)
project(DeepPiTests)

# Set the C++ standard.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build as shared or static library (here we build a static library)
#add_library(DeepPiLib
    #src/YourLibraryFile1.cpp
    #src/YourLibraryFile2.cpp
    # ... add all your library source files here ...
#)

# Enable testing.
enable_testing()

# Find GoogleTest.
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# Create a test executable.
add_executable(DeepPiTests  tests/tensorTests/test_access.cpp)

# Link with GoogleTest and pthread.
target_link_libraries(DeepPiTests GTest::GTest GTest::Main pthread)

# Register the test with CTest.
add_test(NAME DeepPiTests COMMAND DeepPiTests)
