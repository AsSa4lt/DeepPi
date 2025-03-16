#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cstdint>
#include <arm_neon.h>

// A flexible N-dimensional tensor class.
template <typename T, uint16_t N>
class Tensor {
private:
    std::array<uint32_t, N> _strides;  // Strides for converting N indices into a linear index.
    std::vector<T> _data;              // Flat storage for elements.
    std::array<uint32_t, N> _dims;     // Dimensions of the tensor.

    // Compute strides assuming row-major order.
    void computeStrides() {
        _strides[N - 1] = 1;
        for (int i = N - 2; i >= 0; i--) {
            _strides[i] = _dims[i + 1] * _strides[i + 1];
        }
    }

public:
    // Constructor: pass an array with N dimensions.
    Tensor(const std::array<uint32_t, N>& dims) : _dims(dims) {
        uint64_t total = 1;
        for (int i = 0; i < N; i++) {
            total *= _dims[i];
        }
        _data.resize(total);
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
        return _data[linear];
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
        return _data[linear];
    }

    Tensor<T, N> operator+(const Tensor<T, N>& other) const {
        // Check that the dimensions match
        assert(_dims == other._dims && "Tensors must have the same dimensions for addition");

        Tensor<T, N> result(_dims);

        // Loop over the data in chunks (e.g., using 4 elements at a time for floats)
        uint64_t size = _data.size();
        uint64_t i = 0;
        for (; i + 3 < size; i += 4) {
            // Load 4 elements from each tensor
            float32x4_t a = vld1q_f32(&_data[i]);        // Load 4 elements from tensor A
            float32x4_t b = vld1q_f32(&other._data[i]);  // Load 4 elements from tensor B
            
            // Add them
            float32x4_t c = vaddq_f32(a, b);             // Element-wise addition

            // Store the result back to the result tensor
            vst1q_f32(&result._data[i], c);              // Store the result
        }

        // Handle remaining elements if the size is not a multiple of 4
        for (; i < size; ++i) {
            result._data[i] = _data[i] + other._data[i];
        }

        return result;
    }

    
};