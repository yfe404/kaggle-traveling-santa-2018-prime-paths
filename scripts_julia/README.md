
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
# ...
path = nn_opt(path, collect(10:10:length(path)-1), 100); println()                                                                                           
path = nn_opt(path, collect(2:length(path)-1), 25); println()   
```

