# CUDA implementation

To generate the makefile (first time or when changes in CMakeLists.txt):
```bash
mkdir build; cd build
cmake ..
```

Optionally, to specify the build type (default is release ?):
```bash
# Release build (with optimizations)
cmake -DCMAKE_BUILD_TYPE=Release ..
# Debug build (for gdb)
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

To build the project:
```bash
make
```

To run tests:
```bash
make test
```

## Using Docker (to avoid breaking everything with Nvidia drivers)

```bash
docker build -t santa/cuda .
docker run -it -v $(pwd):/src:z santa/cuda bash
cd src/
# Follow instructions above
```
