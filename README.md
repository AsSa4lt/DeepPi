# DeepPi

C++ fundamendtal package for deep learning for scientific computing higly optimized for Raspberry Pi and other ARM devices.

## Installation

### Prerequisites

- C++23 compatible compiler (Clang recommended)
- CMake 3.12+
- For tests: GoogleTest

### Building and Installing

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepPi.git
cd DeepPi

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

## Usage
### CMake
```bash
# In your CMakeLists.txt
find_package(DeepPi REQUIRED)

add_executable(your_project main.cpp)
target_link_libraries(your_project PRIVATE DeepPi::DeepPi)
```

## Comparison with other libraries
We are comparing DeepPi with other libraries like Eigen and Numpy on the same hardware. The benchmarks are done on a Raspberry Pi 4B with 8GB of RAM and a 64-bit OS.

Benchamrk is done for matrix sizes from 128x128 to 2048x2048. The benchmarks are done for matrix multiplication with 10 iterations and the average time is taken. The benchmarks are done for both single precision.

### Comparison with Eigen
Comparing with Eigen, DeepPi is up to 10x faster for big matrices.
![plot](./benchmarks/eigen_vs_deeppi.png)

## Comparison with Numpy
Comparing with Eigen, DeepPi is up 1.25x faster for big matrices, but has bigger overhead for small matrices.
![plot](./benchmarks/numpy_vs_deeppi.png)

## How to install gtest on Ubuntu
```bash
sudo apt-get update
sudo apt-get install libgtest-dev

cd /usr/src/googletest
sudo cmake .
sudo make
sudo cp lib/libgtest*.a /usr/lib/
```

## Running Tests
```bash
# Build the tests
cd build
make test_tensors

# Run the tests
./test_tensors
```

## License
This project is licensed under the MIT License. 
```