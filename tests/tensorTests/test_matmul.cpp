#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <stdexcept>
#include "Tensor/TensorMatmul.h"


// Test case for dotproduct function
TEST(MatMulTests, DotProductValue) {
    // Initialize two simple vectors of size 4
    std::array<uint32_t, 1> dims = {4};
    Tensor<float, 1> A(dims);
    Tensor<float, 1> B(dims);

    A.Data = {1.0f, 2.0f, 3.0f, 4.0f};
    B.Data = {4.0f, 3.0f, 2.0f, 1.0f};

    // Expected result for the dot product: 1*4 + 2*3 + 3*2 + 4*1 = 20
    float expected = 20.0f;

    float result = TensorMatmul::dotproduct(A, B);

    EXPECT_FLOAT_EQ(result, expected);
}

// Test case for dotproduct function
TEST(MatMulTests, DotProductWrongDimensions) {
    // Initialize two simple vectors of size 4
    std::array<uint32_t, 1> dimsA = {4};
    std::array<uint32_t, 1> dimsB = {5};
    Tensor<float, 1> A(dimsA);
    Tensor<float, 1> B(dimsB);

    EXPECT_DEATH({
        TensorMatmul::dotproduct(A, B);
    }, "Vectors must have the same dimensions");
}