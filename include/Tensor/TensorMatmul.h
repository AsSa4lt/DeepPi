#include "Tensor/Tensor.h"
#include <arm_neon.h>
#include <cassert>
#include <cstdint>
#include <array>

namespace TensorMatmul {
    /**
     * @brief Computes the dot product of two single-precision floating point tensors using SIMD operations
     *
     * @param A First input tensor of type Tensor<float, 1>
     * @param B Second input tensor of type Tensor<float, 1>
     * @return The dot product as a float value
     */
    float    dotproduct(const Tensor<float, 1>    &A, const Tensor<float, 1>    &B);

    /**
     * @brief Computes the dot product of two 32-bit unsigned integer tensors tensors using SIMD operations
     *
     * @param A First input tensor of type Tensor<uint32_t, 1>
     * @param B Second input tensor of type Tensor<uint32_t, 1>
     * @return The dot product as a uint32_t value
     */
    uint32_t dotproduct(const Tensor<uint32_t, 1> &A, const Tensor<uint32_t, 1> &B);

    /**
     * @brief Computes the dot product of two 16-bit unsigned integer tensors tensors using SIMD operations
     *
     * @param A First input tensor of type Tensor<uint16_t, 1>
     * @param B Second input tensor of type Tensor<uint16_t, 1>
     * @return The dot product as a uint32_t value
     */
     uint32_t dotproduct(const Tensor<uint16_t, 1> &A, const Tensor<uint16_t, 1> &B);
    
    /**
     * @brief Computes the dot product of two 8-bit unsigned integer tensors tensors using SIMD operations
     *
     * @param A First input tensor of type Tensor<uint8_t, 1>
     * @param B Second input tensor of type Tensor<uint8_t, 1>
     * @return The dot product as a uint32_t value
     */
     uint32_t dotproduct(const Tensor<uint8_t, 1>  &A, const Tensor<uint8_t, 1>  &B);


    /**
     * @brief Computes the dot product of two tensors with unknown or unsupported type
     * Default implementation when we dont know how to handle this type and which SIMD instruction to use
     *
     * @param A First input tensor of type Tensor<T, 1>
     * @param B Second input tensor of type Tensor<T, 1>
     * @return The dot product as a T value
     */
     template <typename T>
     T dotproduct(const Tensor<T, 1>  &A, const Tensor<T, 1>  &B){
        // Check that A and B have the same size
        assert(A.Data.size() == B.Data.size() && "Vectors must have the same dimensions");
        T result = 0; // Initialize the result to zero
        int i = 0;
        for (int i = 0; i < A.Data.size(); i ++) {                     
            result += A(i) * B(i);  
        }
        return result;
     }

    /**
     * @brief Computes the matrix product of two single-precision floating point two-dimensional tensors using SIMD operations
     *
     * @param A First input tensor of type Tensor<float, 2>
     * @param B Second input tensor of type Tensor<float, 2>
     * @return The dot product as a float value
     */
    Tensor<float, 2> matmul2d(const Tensor<float, 2>& A, const Tensor<float, 2>& B);


    template <typename T>
    Tensor<T, 2> matmul2d(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
        static_assert(std::is_same<T, float>::value, "DeepPi only supports float currently.");
        
        const auto& dimsA = A.getDimensions();
        const auto& dimsB = B.getDimensions();
        assert(dimsA[1] == dimsB[0] && "For 2D matrix multiplication, matrices must have shapes MxN and NxK");
    
        uint32_t M_dim = dimsA[0];
        uint32_t N_dim = dimsA[1];
        uint32_t K_dim = dimsB[1];
    
        std::array<uint32_t, 2> dims = {M_dim, K_dim};
        Tensor<T, 2> result(dims);
    
        // Parallelize across rows of result matrix
        for (int i = 0; i < M_dim; i++) {
            for (int j = 0; j < K_dim; j += 4) {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (int k = 0; k < N_dim; k++) {
                    float32x4_t b_vec = vld1q_f32(&B(k, j));
                    float32x4_t a_vec = vdupq_n_f32(A(i, k));
                    sum = vmlaq_f32(sum, a_vec, b_vec);
                }
                vst1q_f32(&result(i, j), sum);
            }
    
            // Handle any remaining columns
            for (int j = (K_dim / 4) * 4; j < K_dim; j++) {
                float sum = 0;
                for (int k = 0; k < N_dim; k++) {
                    sum += A(i, k) * B(k, j);
                }
                result(i, j) = sum;
            }
        }
    
        return result;
    }
};