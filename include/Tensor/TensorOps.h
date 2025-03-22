#pragma once

#include <cstdint>
#include "Tensor/TensorMatmul.h"
#include <Tensor/Tensor.h>
#include <stdexcept>
#include <sys/types.h>
#include <exception>

namespace TensorOps {
    template <typename T, uint16_t N> 
    Tensor<T, N> zeroes(const std::array<uint32_t, N>& dims) {
        Tensor<T, N> result(dims);
        T value = 0;
        result.fillWithValues(value);
        return result;
    }

    template <typename T, uint16_t N> 
    Tensor<T, N> ones(const std::array<uint32_t, N>& dims) {
        Tensor<T, N> result(dims);
        T value = 1;
        result.fillWithValues(value);
        return result;
    }

    template <typename T, uint16_t N>
    Tensor<T, N> full(const std::array<uint32_t, N>& dims, T value) {
        Tensor<T, N> result(dims);
        result.fillWithValues(value);
        return result;
    }

    template <typename T, uint16_t N>
    Tensor<T, N> sum(const Tensor<T,N>& A, const Tensor<T,N>& B){
        return A+B;
    }

    template <typename T, uint16_t N>
    Tensor<T, N> substract(const Tensor<T,N>& A, const Tensor<T,N>& B){
        return A-B;
    }

    template <typename T, uint16_t N_output, uint16_t N_input1, uint16_t N_input2>
    Tensor<T, N_output> matmul(const Tensor<T,N_input1>& A, const Tensor<T,N_input2>& B){
        const auto& dimsA = A.getDimensions();
        const auto& dimsB = B.getDimensions();
        if constexpr(N_input1 == 1 && N_input2 == 1){
            T resultValue = TensorMatmul::dotproduct(A, B);
            std::array<uint32_t, 1> dim = { 1 };
            Tensor<T, N_output> result(dim);
            result(0) = resultValue;
            return result;
        }else if constexpr (N_input1 == N_input2 && N_input1 == 2){
            return TensorMatmul::matmul2d(A, B);
        }

        throw std::logic_error("Not Implemented");
    }

    template <typename T>
    Tensor<T, 2> matmul(const Tensor<T,2>& A, const Tensor<T,2>& B){
        return TensorMatmul::matmul2d(A, B);
    }

    template<typename T>
    T matmul(const Tensor<T,1>& A, const Tensor<T,1>& B){
        return TensorMatmul::dotproduct(A, B);
    }
};