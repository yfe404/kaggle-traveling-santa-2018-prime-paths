
### Install dependencies (first time only)

```bash
julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate()'
julia -e 'import Pkg; Pkg.activate("."); push!(LOAD_PATH, "src/"); using Santa'
```

### K-NN opt

```bash
julia knn_opt.jl -h
# usage: knn_opt.jl --cities FILE --path FILE --knn K [-h]

# optional arguments:
#   --cities FILE  path to kaggle cities.csv file
#   --path FILE    path to the file containing the solution to optimize,
#                  in kaggle .csv submission format
#   --knn K        number of nearest neighbors to consider in the 2-opt
#                  (type: Int64)
#   -h, --help     show this help message and exit
```

```bash
julia -O3 knn_opt.jl --cities ../input/cities.csv --path outputs/sub_knn_opt_1.516828660813334e6.csv --knn 100
# Will output to sub_knn_opt_$score.csv
```

### Parallel K-NN opt

```bash
# For 4 CPUs
julia -O3 -p 4 -L knn_parallel.jl knn_driver.jl
```

Config
```julia
# knn_driver.jl
@everywhere cities_fp = "../input/cities.csv"
@everywhere path_fp = "/home/maxmouchet/Downloads/1516917.csv"
# ...
new_path[chunk] = nn_opt(path[chunk], chunk[1], 1) # <- Change 1 to desired K value
```
