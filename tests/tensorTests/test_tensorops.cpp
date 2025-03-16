#include <gtest/gtest.h>
#include <array>
#include <vector>
#include <stdexcept>
#include "Tensor/TensorOps.h"


TEST(TensorOpsTest, FillZeroes) {
    std::array<uint32_t, 2> dims = {2, 3};
    Tensor<float, 2> tensor(dims);
    tensor.fillWithValues(5.0f);
    EXPECT_FLOAT_EQ(tensor(0, 0), 5.0f);
    tensor = TensorOps::zeroes<float, 2>(dims);    
    EXPECT_FLOAT_EQ(tensor(0, 0), 0.0f);
}

TEST(TensorOpsTest, FillOnes) {
    std::array<uint32_t, 2> dims = {2, 3};
    Tensor<float, 2> tensor(dims);
    tensor.fillWithValues(225.0f);
    EXPECT_FLOAT_EQ(tensor(0, 0), 225.0f);
    tensor = TensorOps::ones<float, 2>(dims);    
    EXPECT_FLOAT_EQ(tensor(0, 0), 1.0f);
}


TEST(TensorOpsTest, SumOps) {
    std::array<uint32_t, 2> dims = {2, 3};
    auto A = TensorOps::full<float, 2>(dims, 6.0f);
    auto B = TensorOps::full<float, 2>(dims, 7.0f);
    auto C = TensorOps::sum(A, B);
    EXPECT_FLOAT_EQ(C(0,0), 13.0f);
    EXPECT_FLOAT_EQ(C(1,2), 13.0f);
}


TEST(TensorOpsTest, SubstractOps) {
    std::array<uint32_t, 2> dims = {2, 3};
    auto A = TensorOps::full<float, 2>(dims, 6.0f);
    auto B = TensorOps::full<float, 2>(dims, 7.0f);
    auto C = TensorOps::substract(A, B);
    EXPECT_FLOAT_EQ(C(0,0), -1.0f);
    EXPECT_FLOAT_EQ(C(1,2), -1.0f);
}