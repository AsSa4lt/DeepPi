#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cstdint>

// A flexible N-dimensional tensor class.
template <typename T, uint16_t N>
class Tensor {
private:
    std::array<uint32_t, N> _dims;     // Dimensions of the tensor.
    std::array<uint32_t, N> _strides;  // Strides for converting N indices into a linear index.
    std::vector<T> _data;              // Flat storage for elements.

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
};