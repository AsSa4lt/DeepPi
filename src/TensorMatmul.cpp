#include "Tensor/TensorOps.h"
#include <arm_neon.h>
#include <cstdint>

/**
* @brief Computes the dot product of two single-precision floating point tensors with SIMD operations
*/
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

/**
* @brief Computes the dot product of two 32-bit unsigned integer tensors tensors with SIMD operations
*/
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

/**
* @brief Computes the dot product of two 16-bit unsigned integer tensors tensors with SIMD operations
*/
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

/**
* @brief Computes the dot product of two 8-bit unsigned integer tensors tensors with SIMD operations
*/
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

Tensor<float, 2> TensorMatmul::naivematmul2d(const Tensor<float, 2>& A, const Tensor<float, 2>& B){
    const auto& dimsA = A.getDimensions();
    const auto& dimsB = B.getDimensions();
    assert(dimsA[1] == dimsB[0] && "For 2D matrix multiplication matrices need to have shapes M*N and N*K");
    uint32_t M_dim = dimsA[0];
    uint32_t N_dim = dimsA[1];
    uint32_t K_dim = dimsB[1];
    std::array<uint32_t, 2> dims = {M_dim, K_dim};
    Tensor<float, 2> result(dims); 
    for(int i = 0; i < M_dim; i++){
        int j = 0;
        for(; j+3 < K_dim; j+=4){ 
            // Initialize sum vector for 4 elements of row i of C
            float32x4_t sum = vdupq_n_f32(0.0f);
            for(int k = 0; k < N_dim; k++){
                // Load A(i,k) and broadcast it into a vector.
                float a_val = A(i, k);
                float32x4_t a_vec = vdupq_n_f32(a_val);
                // Load 4 contiguous floats from row k of B, starting at column j.
                // Since B is row-major, row k starts at index k*K.
                float32x4_t b_vec = vld1q_f32(&B(k,j));
                // Accumulate: sum += a_vec * b_vec
                sum = vmlaq_f32(sum, a_vec, b_vec);
            }
            // Store the computed 4 floats back into matrixC
            vst1q_f32(&result(i, j), sum);
        }
        for(; j < K_dim; j+=1){
            float sum = 0;
            for(int k = 0; k < N_dim; k++){
                sum += A(i,k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Tensor<uint32_t, 2> TensorMatmul::naivematmul2d(const Tensor<uint32_t, 2>& A, const Tensor<uint32_t, 2>& B){
    const auto& dimsA = A.getDimensions();
    const auto& dimsB = B.getDimensions();
    assert(dimsA[1] == dimsB[0] && "For 2D matrix multiplication matrices need to have shapes M*N and N*K");
    uint32_t M_dim = dimsA[0];
    uint32_t N_dim = dimsA[1];
    uint32_t K_dim = dimsB[1];
    std::array<uint32_t, 2> dims = {M_dim, K_dim};
    Tensor<uint32_t, 2> result(dims); 
    for(int i = 0; i < M_dim; i++){
        int j = 0;
        for(; j+3 < K_dim; j+=4){ 
            // Initialize sum vector for 4 elements of row i of C
            uint32x4_t sum = vdupq_n_u32(0.0f);
            for(int k = 0; k < N_dim; k++){
                // Load A(i,k) and broadcast it into a vector.
                uint32_t a_val = A(i, k);
                uint32x4_t a_vec = vdupq_n_u32(a_val);
                // Load 4 contiguous floats from row k of B, starting at column j.
                // Since B is row-major, row k starts at index k*K.
                uint32x4_t b_vec = vld1q_u32(&B(k,j));
                // Accumulate: sum += a_vec * b_vec
                sum = vmlaq_u32(sum, a_vec, b_vec);
            }
            // Store the computed 4 floats back into matrixC
            vst1q_u32(&result(i, j), sum);
        }
        for(; j < K_dim; j+=1){
            uint32_t sum = 0;
            for(int k = 0; k < N_dim; k++){
                sum += A(i,k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}