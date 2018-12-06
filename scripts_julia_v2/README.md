
### Install dependencies (first time only)

```bash
julia -e 'import Pkg; Pkg.activate("."); Pkg.instantiate()'
julia -e 'import Pkg; Pkg.activate("."); push!(LOAD_PATH, "src/"); using Santa'
```
