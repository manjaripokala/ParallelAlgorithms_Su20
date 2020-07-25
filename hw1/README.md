# HW1 - Coding problems

These instructions and code files are just meant to help you get started. You can use these or choose to use your own.
However the final zip file must contain a `hw1.cpp` file which contains an OpenMP implementation of the following
two functions:

`double euclidean_distance(std::vector<double> vector)`
`std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector)`

These instructions are known to work on macOS Mojave. Your mileage may vary in other OSs.

## Pre-requisites:

* OpenMP
* CMake

### Hints on how to install OpenMP:

*MacOs*: `brew install libomp`
*Ubuntu Linux*: `sudo apt install libomp-dev`
*Windows*: Installing in windows is a bit more complex. See these instructions: http://www.mathcancer.org/blog/setting-up-a-64-bit-gcc-environment-on-windows/

### Hints on how to install CMake:

Follow the instruction found here: https://cmake.org/install/

### On macOS use the system's clang instead of XCode's

`export CC=/usr/local/opt/llvm/bin/clang`

`export CXX=/usr/local/opt/llvm/bin/clang++`

`export LDFLAGS="-L/usr/local/opt/llvm/lib"`

`export CPPFLAGS="-I/usr/local/opt/llvm/include"`

## Compiling the project

After having installed OpenMP, perform the following steps, in this directory:

1. `mkdir build/`
2. `cd build`
3. `cmake ../.`
4. `make`

## Executing

Inside the `build` directory, execute:

*`./hw1`

