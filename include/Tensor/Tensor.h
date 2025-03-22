#include <iostream>
#include <type_traits>
#include <vector>
#include <array>
#include <cassert>
#include <cstdint>
#include <arm_neon.h>

#pragma once

// A flexible N-dimensional tensor class.
template <typename T, uint16_t N>
class Tensor {
private:
    std::array<uint32_t, N> _strides;  // Strides for converting N indices into a linear index.
    std::array<uint32_t, N> _dims;     // Dimensions of the tensor.

    // Compute strides assuming row-major order.
    void computeStrides() {
        _strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; i--) {
            _strides[i] = _dims[i + 1] * _strides[i + 1];
        }
    }

public:
    std::vector<T> Data;              // Flat storage for elements.
    static_assert(std::is_floating_point_v<T> || std::is_unsigned_v<T>, "Tensors supports right nor only float");
    
    // Constructor: pass an array with N dimensions.
    Tensor(const std::array<uint32_t, N>& dims) : _dims(dims) {
        uint64_t total = 1;
        for (int i = 0; i < N; i++) {
            total *= _dims[i];
        }
        Data.resize(total);
        computeStrides();
    }

    // Non-const indexing operator: takes exactly N indices.
    template<typename... Index>
    T& operator()(Index... indices) {
        static_assert(sizeof...(indices) == N, "Wrong number of indices");
        std::array<uint32_t, N> idx = { static_cast<uint32_t>(indices)... };
        uint64_t linear = 0;
        for (uint16_t i = 0; i < N; i++) {
            assert(idx[i] < _dims[i] && "Index out of bounds");
            linear += idx[i] * _strides[i];
        }
        return Data[linear];
    }
    // Const indexing operator.
    template<typename... Index>
    const T& operator()(Index... indices) const {
        static_assert(sizeof...(indices) == N, "Wrong number of indices");
        std::array<uint32_t, N> idx = { static_cast<uint32_t>(indices)... };
        uint64_t linear = 0;
        for (uint16_t i = 0; i < N; i++) {
            assert(idx[i] < _dims[i] && "Index out of bounds");
            linear += idx[i] * _strides[i];
        }
        return Data[linear];
    }


    void fillWithValues(uint32_t value){
        int i = 0;
        for(; i + 3 < Data.size(); i+=4){
            uint32x4_t vec1 = vdupq_n_u32(value);
            vst1q_u32(&Data[i], vec1);
        }
        for(;i < Data.size(); i++){
            Data[i] = value;
        }
    }

    void fillWithValues(uint16_t value){
        int i = 0;
        for(; i + 7 < Data.size(); i+=8){
            uint16x8_t vec1 = vdupq_n_u16(value);
            vst1q_u16(&Data[i], vec1);
        }
        for(;i < Data.size(); i++){
            Data[i] = value;
        }
    }


    void fillWithValues(uint8_t value){
        int i = 0;
        for(; i + 7 < Data.size(); i+=8){
            uint8x16_t vec1 = vdupq_n_u8(value);
            vst1q_u8(&Data[i], vec1);
        }
        for(;i < Data.size(); i++){
            Data[i] = value;
        }
    }


    void fillWithValues(float value){
        int i = 0;
        for(; i + 3 < Data.size(); i+=4){
            float32x4_t vec1 = vdupq_n_f32(value);
            vst1q_f32(&Data[i], vec1);
        }

        for(;i < Data.size(); i++){
            Data[i] = value;
        }
    }
    
    const std::array<uint32_t, N>& getDimensions() const{
        return _dims;
    }



    Tensor<uint32_t, N> operator+(const Tensor<uint32_t, N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");
        Tensor<uint32_t, N> result(_dims);
        uint64_t i = 0;
        for (; i + 3 < Data.size(); i += 4) {          // Loop over in chunks of 4
            uint32x4_t a = vld1q_u32(&Data[i]);        // Load 4 elements from tensor A
            uint32x4_t b = vld1q_u32(&other.Data[i]);  // Load 4 elements from tensor B
            uint32x4_t c = vaddq_u32(a, b);     // Element-wise addition
            vst1q_u32(&result.Data[i], c);             // Store the result
        }
        for (; i < Data.size(); ++i) {                 // Handle remaining elements if the size is not a multiple of 4
            result.Data[i] = Data[i] + other.Data[i];
        }
        return result;
    }

    Tensor<uint16_t, N> operator+(const Tensor<uint16_t, N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");
        Tensor<uint16_t, N> result(_dims);
        uint64_t i = 0;
        for (; i + 7 < Data.size(); i += 8) {          // Loop over in chunks of 8
            uint16x8_t a = vld1q_u16(&Data[i]);        // Load 8 elements from tensor A
            uint16x8_t b = vld1q_u16(&other.Data[i]);  // Load 8 elements from tensor B
            uint16x8_t c = vaddq_u16(a, b);     // Element-wise addition
            vst1q_u16(&result.Data[i], c);             // Store the result
        }
        for (; i < Data.size(); ++i) {                 // Handle remaining elements if the size is not a multiple of 8
            result.Data[i] = Data[i] + other.Data[i];
        }
        return result;
    }

    Tensor<uint8_t, N> operator+(const Tensor<uint8_t, N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");
        Tensor<uint8_t, N> result(_dims);
        uint64_t i = 0;
        for (; i + 15 < Data.size(); i += 16) {       // Loop over in chunks of 16
            uint8x16_t a = vld1q_u8(&Data[i]);        // Load 16 elements from tensor A
            uint8x16_t b = vld1q_u8(&other.Data[i]);  // Load 16 elements from tensor B
            uint8x16_t c = vaddq_u8(a, b);     // Element-wise addition
            vst1q_u8(&result.Data[i], c);             // Store the result
        }
        for (; i < Data.size(); ++i) {                // Handle remaining elements if the size is not a multiple of 16
            result.Data[i] = Data[i] + other.Data[i];
        }
        return result;
    }

    Tensor<float, N> operator+(const Tensor<float, N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");
        Tensor<float, N> result(_dims);
        uint64_t i = 0;
        for (; i + 3 < Data.size(); i += 4) {           // Loop over in chunks of 4
            float32x4_t a = vld1q_f32(&Data[i]);        // Load 4 elements from tensor A
            float32x4_t b = vld1q_f32(&other.Data[i]);  // Load 4 elements from tensor B
            float32x4_t c = vaddq_f32(a, b);     // Element-wise addition
            vst1q_f32(&result.Data[i], c);              // Store the result
        }
        for (; i < Data.size(); ++i) {                  // Handle remaining elements if the size is not a multiple of 4
            result.Data[i] = Data[i] + other.Data[i];
        }
        return result;
    }


    Tensor<uint32_t, N> operator-(const Tensor<uint32_t,N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for substraction");
        Tensor<uint32_t, N> result(_dims);
        uint64_t size = Data.size();
        uint64_t i = 0;
        for (; i + 3 < size; i += 4) {
            uint32x4_t a = vld1q_u32(&Data[i]);         // Load 4 elements from tensor A
            uint32x4_t b = vld1q_u32(&other.Data[i]);   // Load 4 elements from tensor B
            uint32x4_t c = vsubq_u32(a, b);      // Element-wise substraction
            vst1q_u32(&result.Data[i], c);              // Store the result
        }
        for (; i < size; ++i) {                         // Handle remaining elements if the size is not a multiple of 4
            result.Data[i] = Data[i] - other.Data[i];
        }
        return result;
    }

    Tensor<uint16_t, N> operator-(const Tensor<uint16_t, N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");
        Tensor<uint16_t, N> result(_dims);
        uint64_t i = 0;
        for (; i + 7 < Data.size(); i += 8) {          // Loop over in chunks of 8
            uint16x8_t a = vld1q_u16(&Data[i]);        // Load 8 elements from tensor A
            uint16x8_t b = vld1q_u16(&other.Data[i]);  // Load 8 elements from tensor B
            uint16x8_t c = vsubq_u16(a, b);     // Element-wise substraction
            vst1q_u16(&result.Data[i], c);             // Store the result
        }
        for (; i < Data.size(); ++i) {                 // Handle remaining elements if the size is not a multiple of 8
            result.Data[i] = Data[i] - other.Data[i];
        }
        return result;
    }

    Tensor<uint8_t, N> operator-(const Tensor<uint8_t, N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");
        Tensor<uint8_t, N> result(_dims);
        uint64_t i = 0;
        for (; i + 15 < Data.size(); i += 16) {       // Loop over in chunks of 16
            uint8x16_t a = vld1q_u8(&Data[i]);        // Load 16 elements from tensor A
            uint8x16_t b = vld1q_u8(&other.Data[i]);  // Load 16 elements from tensor B
            uint8x16_t c = vsubq_u8(a, b);     // Element-wise substraction
            vst1q_u8(&result.Data[i], c);             // Store the result
        }
        for (; i < Data.size(); ++i) {                // Handle remaining elements if the size is not a multiple of 16
            result.Data[i] = Data[i] - other.Data[i];
        }
        return result;
    }

    Tensor<float, N> operator-(const Tensor<float,N>& other) const {
        assert(_dims == other._dims && "Tensors must have the same dimensions for substraction");
        Tensor<float, N> result(_dims);
        uint64_t size = Data.size();
        uint64_t i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t a = vld1q_f32(&Data[i]);        // Load 4 elements from tensor A
            float32x4_t b = vld1q_f32(&other.Data[i]);  // Load 4 elements from tensor B
            float32x4_t c = vsubq_f32(a, b);     // Element-wise substraction
            vst1q_f32(&result.Data[i], c);              // Store the result
        }
        for (; i < size; ++i) {                         // Handle remaining elements if the size is not a multiple of 4
            result.Data[i] = Data[i] - other.Data[i];
        }
        return result;
    }

    
};