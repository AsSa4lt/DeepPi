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
        result.fillWithValues(0);
        return result;
    }

    template <typename T, uint16_t N> 
    Tensor<T, N> ones(const std::array<uint32_t, N>& dims) {
        Tensor<T, N> result(dims);
        result.fillWithValues(1);
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

    template <typename T, uint16_t N>
    Tensor<T, N> matmul(const Tensor<T,N>& A, const Tensor<T,N>& B){
        const auto& dimsA = A.getDimensions();
        const auto& dimsB = B.getDimensions();
        if(N == 1){
            T resultValue = TensorMatmul::dotproduct(A, B);
            std::array<uint32_t, 4> dim = { 1 };
            Tensor<T, N> result(dim);
            result(0) = resultValue;
            return result;
        }

        throw std::logic_error("Not Implemented");
    }
};