#include "Tensor/TensorOps.h"
#include <arm_neon.h>
#include <cstdint>

float TensorMatmul::dotproduct(const Tensor<float, 1> &A, const Tensor<float, 1> &B){
    // Check that A and B have the same size
    assert(A.Data.size() == B.Data.size() && "Vectors must have the same dimensions");
    float result = 0.0f; // Initialize the result to zero
    int i = 0;
    for (; i + 3 < A.Data.size(); i += 4) {
        // Load 4 elements from tensor A and tensor B
        float32x4_t a = vld1q_f32(&A(i));
        float32x4_t b = vld1q_f32(&B(i));
        float32x4_t prod = vmulq_f32(a, b);
        // Perform a horizontal addition to sum the 4 values in the prod vector
        // Add the low and high parts
        float32x2_t sum = vadd_f32(vget_low_f32(prod), vget_high_f32(prod)); 
        // Horizontal add the two elements
        sum = vpadd_f32(sum, sum);
        // Extract the sum from the 2-element vector                     
        result += vget_lane_f32(sum, 0);  
    }
    for (; i < A.Data.size(); ++i) {
        result += A(i) * B(i);
    }
    return result;
}

uint32_t TensorMatmul::dotproduct(const Tensor<uint32_t, 1> &A, const Tensor<uint32_t, 1> &B){
    // Check that A and B have the same size
    assert(A.Data.size() == B.Data.size() && "Vectors must have the same dimensions");
    uint32_t result = 0; // Initialize the result to zero
    int i = 0;
    for (; i + 3 < A.Data.size(); i += 4) {
        // Load 4 elements from tensor A and tensor B
        uint32x4_t a = vld1q_u32(&A(i));
        uint32x4_t b = vld1q_u32(&B(i));
        uint32x4_t prod = vmulq_u32(a, b);
        // Perform a horizontal addition to sum the 4 values in the prod vector
        // Add the low and high parts
        uint32x2_t sum = vadd_u32(vget_low_u32(prod), vget_high_u32(prod)); 
        // Horizontal add the two elements
        sum = vpadd_u32(sum, sum);
        // Extract the sum from the 2-element vector                     
        result += vget_lane_u32(sum, 0);  
    }
    for (; i < A.Data.size(); ++i) {
        result += A(i) * B(i);
    }
    return result;
}

uint32_t TensorMatmul::dotproduct(const Tensor<uint16_t, 1> &A, const Tensor<uint16_t, 1> &B){
    // Check that A and B have the same size
    assert(A.Data.size() == B.Data.size() && "Vectors must have the same dimensions");
    uint32_t result = 0; // Initialize the result to zero
    int i = 0;
    for (; i + 7 < A.Data.size(); i += 8) {
        // Load 8 elements from tensor A and tensor B
        uint16x8_t a = vld1q_u16(&A(i));
        uint16x8_t b = vld1q_u16(&B(i));
        uint16x8_t prod = vmulq_u32(a, b);
        // Perform a horizontal addition to sum the 8 values in the prod vector
        // Add the low and high parts
        uint16x4_t sum = vadd_u16(vget_low_u16(prod), vget_high_u16(prod));  
        // Horizontal add the two elements
        sum = vpadd_u16(sum, sum);
        // Extract the sum from the 4-element vector                     
        result += vget_lane_u16(sum, 0);  
    }
    for (; i < A.Data.size(); ++i) {
        result += A(i) * B(i);
    }
    return result;
}

uint32_t TensorMatmul::dotproduct(const Tensor<uint8_t, 1> &A, const Tensor<uint8_t, 1> &B){
    // Check that A and B have the same size
    assert(A.Data.size() == B.Data.size() && "Vectors must have the same dimensions");
    uint32_t result = 0; // Initialize the result to zero
    int i = 0;
    for (; i + 15 < A.Data.size(); i += 16) {
        // Load 4 elements from tensor A and tensor B
        uint8x16_t a = vld1q_u8(&A(i));
        uint8x16_t b = vld1q_u8(&B(i));
        uint8x16_t prod = vmulq_u8(a, b);
        // Perform a horizontal addition to sum the 16 values in the prod vector
        // Add the low and high parts
        uint8x8_t low = vget_low_u8(prod);
        uint8x8_t high = vget_high_u8(prod);
        // Horizontal add the two elements
        uint8x8_t sum = vpadd_u8(low, high);
        // Extract the sum from the 8-element vector                     
        result += vget_lane_u8(sum, 0);  
    }
    for (; i < A.Data.size(); ++i) {
        result += A(i) * B(i);
    }
    return result;
}