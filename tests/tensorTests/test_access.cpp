#include <gtest/gtest.h>
#include "Tensor.h" 

// Test that valid access works correctly.
TEST(TensorTest, ValidAccess) {
    // Create a 4D tensor with dimensions 2 x 3 x 4 x 5.
    std::array<uint32_t, 4> dims = {2, 3, 4, 5};
    Tensor<float, 4> tensor(dims);

    // Set a value at a valid index.
    tensor(1, 2, 3, 4) = 3.14f;

    // Verify that the value is stored and retrieved correctly.
    EXPECT_FLOAT_EQ(tensor(1, 2, 3, 4), 3.14f);
}

// Test that accessing an invalid index triggers an assert failure.
// This is a death test, so it will only work when assertions are enabled.
TEST(TensorTest, OutOfBoundsAccess) {
    std::array<uint32_t, 4> dims = {2, 3, 4, 5};
    Tensor<float, 4> tensor(dims);

    // Attempt to access an out-of-bound index.
    // For example, for the first dimension, valid indices are 0 and 1.
    EXPECT_DEATH({
        float value = tensor(2, 0, 0, 0);  // index '2' is out-of-bounds
        (void)value;  // silence unused variable warning
    }, "Index out of bounds");
}

// Optional: Provide a main() if you don't use gtest_main.
// If you link against gtest_main in your CMakeLists, you don't need this.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
