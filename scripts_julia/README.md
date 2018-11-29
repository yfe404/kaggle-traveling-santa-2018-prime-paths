
### Install dependencies (first time only)

```bash
julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate()'
```

### K-NN opt

```bash
julia -O3 knn_opt.jl ../input/cities.csv kaggle_submission_file.csv
# Will output to sub_knn_opt_$score.csv
```

Config
```julia
# Edit the end of knn_opt.jl
path = nn_opt(path, collect(10:10:length(path)-1), 100); println()                                                                                           
path = nn_opt(path, collect(2:length(path)-1), 25); println()   
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
