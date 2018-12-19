# CUDA implementation

## Overview

- Generic code for coordinates to avoid casts
- Kaggle CSV in (`read_path`) / Kaggle CSV out (`write_path`)

The main data structure is `City`:
```cpp
struct City {
    int i;
    bool p; // Whether i is prime or not
    Coord<T> xy;
    // ...
};
```

A path is an array/vector/... of `City`.

## Requirements

- [Cuda](https://developer.nvidia.com/cuda-downloads) 10
- [CMake](https://cmake.org/download/) >= 3.8 
- [CxxTest](http://cxxtest.com/) (For testing only, can be compiled without)

Note: CMake 3.8 is not available in the prehistoric Debian repositories of the GCE Intel DL image.  
So: compile it from source (works, but slow), or use a less prehistoric distribution (such as Ubuntu).

## Build

To generate the makefile (first time or when changes in CMakeLists.txt):
```bash
mkdir build; cd build
cmake ..
```

Optionally, to specify the build type (default is release):
```bash
# Debug build (for gdb)
cmake -DCMAKE_BUILD_TYPE=Debug ..
# Release build (with optimizations)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

To build the project:
```bash
make
```

To run tests:
```bash
make test
```

### Using Docker (to avoid breaking everything with Nvidia drivers)

```bash
docker build -t santa/cuda .
docker run -it -v $(pwd):/src:z santa/cuda bash
cd src/
# Follow instructions above
```
