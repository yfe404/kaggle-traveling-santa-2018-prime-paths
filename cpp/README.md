

### Build

```bash
mkdir build; cd build
cmake ..
make
```

```bash
# Release (default ?) build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Debug build (for gdb)
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```
