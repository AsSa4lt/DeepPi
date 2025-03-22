#include <cstdint>
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <stdexcept>
#include "Tensor/TensorOps.h"


// Test case for dotproduct function
TEST(MatmulTests, DotProductValue) {
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
TEST(MatmulTests, DotProductWrongDimensions) {
    // Initialize two simple vectors of size 4
    std::array<uint32_t, 1> dimsA = {4};
    std::array<uint32_t, 1> dimsB = {5};
    Tensor<float, 1> A(dimsA);
    Tensor<float, 1> B(dimsB);

    EXPECT_DEATH({
        TensorMatmul::dotproduct(A, B);
    }, "Vectors must have the same dimensions");
}

TEST(MatmulTests, Matmul2DValueSimple){
    std::array<uint32_t, 2> dims = {2, 2};
    auto A = TensorOps::full<float, 2>(dims, 6.0f);
    auto B = TensorOps::full<float, 2>(dims, 7.0f);
    Tensor<float, 2> result = TensorMatmul::matmul2d(A, B);
    EXPECT_FLOAT_EQ(84.0f, result(0,0));
}

TEST(MatmulTests, Matmul2DDifferentMK){
    std::array<uint32_t, 2> dimsA = {3, 5};
    std::array<uint32_t, 2> dimsB = {5, 6};
    auto A = TensorOps::full<float, 2>(dimsA, 11.0f);
    auto B = TensorOps::full<float, 2>(dimsB, 23.0f);
    Tensor<float, 2> result = TensorMatmul::matmul2d(A, B);
    // Tenst dimensions
    std::array<uint32_t, 2> dimsC = result.getDimensions();
    EXPECT_EQ(3, dimsC[0]);
    EXPECT_EQ(6, dimsC[1]);
    // test value
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 6; j++){
            EXPECT_FLOAT_EQ(1265.0f, result(i,j));
        }
    }
}

TEST(MatmulTests, Matmul2DWrongDimenstion){
    std::array<uint32_t, 2> dimsA = {3, 1};
    std::array<uint32_t, 2> dimsB = {5, 6};
    auto A = TensorOps::full<float, 2>(dimsA, 11.0f);
    auto B = TensorOps::full<float, 2>(dimsB, 23.0f);
    EXPECT_DEATH({
        TensorMatmul::matmul2d(A, B);
    }, "need to have shapes");
}

TEST(MatmulTests, MatmullCallVectorsMainFunc){
    // Initialize two simple vectors of size 4
    std::array<uint32_t, 1> dims = {4};
    Tensor<float, 1> A(dims);
    Tensor<float, 1> B(dims);
    A.Data = {1.0f, 2.0f, 3.0f, 4.0f};
    B.Data = {4.0f, 3.0f, 2.0f, 1.0f};
    float expected = 20.0f;
    Tensor<float, 1> result = TensorOps::matmul<float, 1, 1, 1>(A, B);
    EXPECT_FLOAT_EQ(result(0), expected);
}

TEST(MatmulTests, MatmullCallVectors){
    // Initialize two simple vectors of size 4
    std::array<uint32_t, 1> dims = {4};
    Tensor<float, 1> A(dims);
    Tensor<float, 1> B(dims);
    A.Data = {1.0f, 2.0f, 3.0f, 4.0f};
    B.Data = {4.0f, 3.0f, 2.0f, 1.0f};
    float expected = 20.0f;
    float result = TensorOps::matmul<float>(A, B);
    EXPECT_FLOAT_EQ(result, expected);
}

TEST(MatmulTests, MatmullCall2DMatrices){
    std::array<uint32_t, 2> dimsA = {3, 5};
    std::array<uint32_t, 2> dimsB = {5, 6};
    auto A = TensorOps::full<float, 2>(dimsA, 11.0f);
    auto B = TensorOps::full<float, 2>(dimsB, 23.0f);
    Tensor<float, 2> result = TensorOps::matmul<float, 2, 2, 2>(A, B);
    // Test dimensions
    std::array<uint32_t, 2> dimsC = result.getDimensions();
    EXPECT_EQ(3, dimsC[0]);
    EXPECT_EQ(6, dimsC[1]);
    // test value
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 6; j++){
            EXPECT_FLOAT_EQ(1265.0f, result(i,j));
        }
    }
}

TEST(MatmulTests, MatmullCall2DMatricesMainFunc){
    std::array<uint32_t, 2> dimsA = {3, 5};
    std::array<uint32_t, 2> dimsB = {5, 6};
    auto A = TensorOps::full<float, 2>(dimsA, 11.0f);
    auto B = TensorOps::full<float, 2>(dimsB, 23.0f);
    Tensor<float, 2> result = TensorOps::matmul<float, 2, 2, 2>(A, B);
    // Test dimensions
    std::array<uint32_t, 2> dimsC = result.getDimensions();
    dimsC = result.getDimensions();
    EXPECT_EQ(3, dimsC[0]);
    EXPECT_EQ(6, dimsC[1]);
    // test value
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 6; j++){
            EXPECT_FLOAT_EQ(1265.0f, result(i,j));
        }
    }
}