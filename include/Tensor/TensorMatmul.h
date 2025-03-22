#include "Tensor/Tensor.h"
#include <arm_neon.h>
#include <cassert>
#include <cstdint>
#include <array>

namespace TensorMatmul {
    float    dotproduct(const Tensor<float, 1>    &A, const Tensor<float, 1>    &B);
    uint32_t dotproduct(const Tensor<uint32_t, 1> &A, const Tensor<uint32_t, 1> &B);
    uint32_t dotproduct(const Tensor<uint16_t, 1> &A, const Tensor<uint16_t, 1> &B);
    uint32_t dotproduct(const Tensor<uint8_t, 1>  &A, const Tensor<uint8_t, 1>  &B);

    template <typename T, uint16_t N>
    Tensor<T, N> matmul2d(const Tensor<T, N>& A, const Tensor<T, N>& B){
        assert(N == 2 && "For 2D matrix multiplication tensor need to be 2-dimensional");
        static_assert(std::is_same<T, float>::value, "DeepPi only supports float currently.");
        const auto& dimsA = A.getDimensions();
        const auto& dimsB = B.getDimensions();
        assert(dimsA[1] == dimsB[0] && "For 2D matrix multiplication matrices need to have shapes M*N and N*K");
        uint32_t M_dim = dimsA[0];
        uint32_t N_dim = dimsA[1];
        uint32_t K_dim = dimsB[1];
        std::array<uint32_t, 2> dims = {M_dim, K_dim};
        Tensor<T, N> result(dims); 
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
};