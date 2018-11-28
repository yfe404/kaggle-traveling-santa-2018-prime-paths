
```
UInt32[0xfd739d89, 0xdf3824c4, 0x78037b18, 0x7e10b0f5]
2006.0 iterations/s] 7324700/10000000 score = 1.4184348143132916e8
```

```bash
julia -O3 --check-bounds=no --math-mode=fast swapper.jl ../../cities.csv ../scripts/genetic_pool/1517028.csv 10000000
```
