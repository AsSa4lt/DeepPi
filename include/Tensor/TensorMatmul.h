#include "Tensor/Tensor.h"
#include <cassert>

namespace TensorMatmul {
    template <typename T, uint16_t N>
    T dotproduct(const Tensor<T, N>& A, const Tensor<T, N>& B) {
        // Check that A and B have the same size
        static_assert(N == 1 && "For dotproduct you must multiply vectors");
        assert(A.Data.size() == B.Data.size() && "Vectors must have the same dimensions");
    
        T result = 0.0f; // Initialize the result to zero
        int i = 0;
    
        // Loop over data in chunks of 4
        for (; i + 3 < A.Data.size(); i += 4) {
            // Load 4 elements from each tensor A and B
            float32x4_t a = vld1q_f32(&A.Data[i]);        // Load 4 elements from tensor A
            float32x4_t b = vld1q_f32(&B.Data[i]);        // Load 4 elements from tensor B
    
            // Multiply corresponding elements of A and B
            float32x4_t prod = vmulq_f32(a, b);           // Element-wise multiplication
    
            // Perform a horizontal addition to sum the 4 values in the prod vector
            float32x2_t sum = vadd_f32(vget_low_f32(prod), vget_high_f32(prod)); // Add the low and high parts
            sum = vpadd_f32(sum, sum);                     // Horizontal add the two elements
    
            // Accumulate the result
            result += vget_lane_f32(sum, 0);              // Extract the sum from the 2-element vector
        }
    
        // Handle any remaining elements (less than 4 elements left)
        for (; i < A.Data.size(); ++i) {
            result += A.Data[i] * B.Data[i];
        }
    
        return result;
    }
};