#include <cstdint>
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <stdexcept>
#include "Tensor/Tensor.h" 
#include "Tensor/TensorOps.h"

// Test 1 - Basic Sum Test
TEST(TensorSumTest, BasicSumFloat) {
    std::array<uint32_t, 2> dims = {2, 3};
    
    Tensor<float, 2> A(dims);
    Tensor<float, 2> B(dims);

    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(0, 2) = 3.0f;
    A(1, 0) = 4.0f; A(1, 1) = 5.0f; A(1, 2) = 6.0f;

    B(0, 0) = 7.0f; B(0, 1) = 8.0f; B(0, 2) = 9.0f;
    B(1, 0) = 10.0f; B(1, 1) = 11.0f; B(1, 2) = 12.0f;

    // Perform the sum
    Tensor<float, 2> C = A + B;

    // Verify results
    EXPECT_FLOAT_EQ(C(0, 0), 8.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(C(0, 2), 12.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 14.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 16.0f);
    EXPECT_FLOAT_EQ(C(1, 2), 18.0f);
}


// Test 2 - Adding Tensors with Different Sizes (Expecting Failure)
TEST(TensorSumTest, SumDifferentSizesFloat) {
    std::array<uint32_t, 2> dimsA = {3, 3};
    std::array<uint32_t, 2> dimsB = {2, 3};
    
    Tensor<float, 2> A(dimsA);
    Tensor<float, 2> B(dimsB);

    EXPECT_DEATH({
        A + B;
    },  "Tensors must have the same dimensions for addition");
}

// Test 3 - Edge Case: Zero-Sized Tensors
TEST(TensorSumTest, SumZeroSizedTensorFloat) {
    std::array<uint32_t, 2> dims = {0, 0};
    
    Tensor<float, 2> A(dims);
    Tensor<float, 2> B(dims);

    Tensor<float, 2> C = A + B;
}

// Test 4 - Performance
TEST(TensorSumTest, SumPerformanceTestFloat) {
    std::array<uint32_t, 2> dims = {2048, 2048};  // Large tensor
    
    Tensor<float, 2> A(dims);
    Tensor<float, 2> B(dims);

    for (size_t i = 0; i < 2048; ++i) {
        for (size_t j = 0; j < 2048; ++j) {
            A(i, j) = 1.0f;
            B(i, j) = 2.0f;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    Tensor<float, 2> C = A + B;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Tensor sum performance test took " << duration.count() << " ms" << std::endl;
}

// Test 5 Test for uint32
TEST(TensorSumTest, BasicSumUint32) {
    std::array<uint32_t, 2> dims = {2, 3};
    Tensor<uint32_t, 2> A = TensorOps::full<uint32_t, 2>(dims, 5);
    Tensor<uint32_t, 2> B = TensorOps::full<uint32_t, 2>(dims, 7);
    // set one number to be different
    A(0, 2) = 11;
    // Perform the sum
    Tensor<uint32_t, 2> C = A + B;
    // Verify results
    EXPECT_EQ(12, C(0,0));
    EXPECT_EQ(18, C(0,2));
}

// Test 6 Test for uint16
TEST(TensorSumTest, BasicSumUint16) {
    std::array<uint32_t, 2> dims = {2, 3};
    Tensor<uint16_t, 2> A = TensorOps::full<uint16_t, 2>(dims, 5);
    Tensor<uint16_t, 2> B = TensorOps::full<uint16_t, 2>(dims, 7);
    // set one number to be different
    A(0, 2) = 11;
    // Perform the sum
    Tensor<uint16_t, 2> C = A + B;
    // Verify results
    EXPECT_EQ(12, C(0,0));
    EXPECT_EQ(18, C(0,2));
}

// Test 7 Test for uint16
TEST(TensorSumTest, BasicSumUint8) {
    std::array<uint32_t, 2> dims = {2, 3};
    Tensor<uint8_t, 2> A = TensorOps::full<uint8_t, 2>(dims, 5);
    Tensor<uint8_t, 2> B = TensorOps::full<uint8_t, 2>(dims, 7);
    // set one number to be different
    A(0, 2) = 11;
    // Perform the sum
    Tensor<uint8_t, 2> C = A + B;
    // Verify results
    EXPECT_EQ(12, C(0,0));
    EXPECT_EQ(18, C(0,2));
}
