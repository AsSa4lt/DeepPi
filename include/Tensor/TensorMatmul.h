#include "Tensor/Tensor.h"
#include <arm_neon.h>
#include <cassert>
#include <cstdint>
#include <array>
#include <future>

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
     * @return The matrix multiplication product as a Tensor<float, 2> value
     */
    Tensor<float, 2> naivematmul2d(const Tensor<float, 2>& A, const Tensor<float, 2>& B);


    /**
     * @brief Internal function for matrix multiplications using impoved Strassen algorithm
     * 
     * @param A First input tensor of type Tensor<T, 2>
     * @param B Second input tensor of type Tensor<T, 2>
     * @return The matrix multiplication product as a Tensor<T, 2> value
     */
    template <typename T>
    Tensor<T, 2> matmul2dStrassen(const Tensor<T, 2>& A, const Tensor<T, 2>& B, int level){
        const auto& dimsA = A.getDimensions();
        const auto& dimsB = B.getDimensions();
        uint32_t M_dim = dimsA[0];
        uint32_t N_dim = dimsA[1];
        uint32_t K_dim = dimsB[1];
        // if we can't use Winograds algorithm
        if (dimsA[0] < 2 || dimsB[0] < 2 || dimsB[1] < 2)
            return naivematmul2d(A, B);
        // Or if matrices are small
        if (dimsA[0] * dimsA[1] * dimsB[1] < 512)
            return naivematmul2d(A, B);

        std::array<uint32_t, 2> dims = {M_dim, K_dim};
        Tensor<T, 2> result{dims};
        // Set "correct matrices"
        Tensor<T, 2> A_div2 = A;
        Tensor<T, 2> B_div2 = B;

        if (dimsA[0] % 2 == 1 || dimsA[1] % 2 == 1 || dimsB[1] % 2 == 1){
            result = result.ExtendToDivisibleBy2();
            A_div2 = A_div2.ExtendToDivisibleBy2();
            B_div2 = B_div2.ExtendToDivisibleBy2();
        }
        

        Tensor<T, 2> A11 = A_div2.LeftTopPart();
        Tensor<T, 2> A12 = A_div2.RightTopPart();
        Tensor<T, 2> A21 = A_div2.LeftBottomPart();
        Tensor<T, 2> A22 = A_div2.RightBottomPart();
        Tensor<T, 2> B11 = B_div2.LeftTopPart();
        Tensor<T, 2> B12 = B_div2.RightTopPart();
        Tensor<T, 2> B21 = B_div2.LeftBottomPart();
        Tensor<T, 2> B22 = B_div2.RightBottomPart();

        
        // Compute M1 to M7
        if (level == 0){
            auto futureM1 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A11 + A22, B11 + B22, level++); });
            auto futureM2 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A21 + A22, B11, level++); });
            auto futureM3 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A11, B12 - B22, level++); });
            auto futureM4 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A22, B21 - B11, level++); });
            auto futureM5 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A11 + A12, B22, level++); });
            auto futureM6 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A21 - A11, B11 + B12, level++); });
            auto futureM7 = std::async(std::launch::async, [&]() { return matmul2dStrassen(A12 - A22, B21 + B22, level++); });
        
            // Get the results from the futures
            Tensor<T, 2> M1 = futureM1.get();
            Tensor<T, 2> M2 = futureM2.get();
            Tensor<T, 2> M3 = futureM3.get();
            Tensor<T, 2> M4 = futureM4.get();
            Tensor<T, 2> M5 = futureM5.get();
            Tensor<T, 2> M6 = futureM6.get();
            Tensor<T, 2> M7 = futureM7.get();

            // Compute final submatrices of the result
            Tensor<T, 2> C11 = M1 + M4 - M5 + M7;
            Tensor<T, 2> C12 = M3 + M5;
            Tensor<T, 2> C21 = M2 + M4;
            Tensor<T, 2> C22 = M1 - M2 + M3 + M6;

            // Combine the submatrices into the final result
            result.FillLeftTopPart(C11);
            result.FillRightTopPart(C12);
            result.FillLeftBottomPart(C21);
            result.FillRightBottomPart(C22);

            result = result.CutToDimensions(M_dim, K_dim);
            return result;
        }else {
            // Compute M1 to M7
            Tensor<T, 2> M1 = matmul2dStrassen(A11 + A22, B11 + B22, level++);
            Tensor<T, 2> M2 = matmul2dStrassen(A21 + A22, B11, level++);
            Tensor<T, 2> M3 = matmul2dStrassen(A11, B12 - B22, level++);
            Tensor<T, 2> M4 = matmul2dStrassen(A22, B21 - B11, level++);
            Tensor<T, 2> M5 = matmul2dStrassen(A11 + A12, B22, level++);
            Tensor<T, 2> M6 = matmul2dStrassen(A21 - A11, B11 + B12, level++);
            Tensor<T, 2> M7 = matmul2dStrassen(A12 - A22, B21 + B22, level++);

            // Compute final submatrices of the result
            Tensor<T, 2> C11 = M1 + M4 - M5 + M7;
            Tensor<T, 2> C12 = M3 + M5;
            Tensor<T, 2> C21 = M2 + M4;
            Tensor<T, 2> C22 = M1 - M2 + M3 + M6;

            // Combine the submatrices into the final result
            result.FillLeftTopPart(C11);
            result.FillRightTopPart(C12);
            result.FillLeftBottomPart(C21);
            result.FillRightBottomPart(C22);

            result = result.CutToDimensions(M_dim, K_dim);
            return result;
        }
    }
     
    /**
     * @brief Computes the matrix product of two two-dimensional tensors with unknown type
     * It's just a wrapper for a real function
     *
     * @param A First input tensor of type Tensor<T, 2>
     * @param B Second input tensor of type Tensor<T, 2>
     * @return The matrix multiplication product as a Tensor<T, 2> value
     */
    template <typename T>
    Tensor<T, 2> matmul2d(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
        static_assert(std::is_same<T, float>::value, "DeepPi only supports float currently.");
        return matmul2dStrassen(A, B, 0);
    }
};