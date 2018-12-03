#!/usr/bin/env julia
git_root = strip(read(`git rev-parse --show-toplevel`, String))
base_dir = joinpath(git_root, "scripts_julia")

println("Running from $(pwd())")
println("Base dir is $(base_dir)")

push!(LOAD_PATH, joinpath(base_dir, "src/"))
import Pkg; Pkg.activate(base_dir)
using Santa

# OPTIMIZE: In this case we could precompute/cache closest cities since the path in chunk is not altered
function nn_opt(init_path::Vector{City}, start::Int, stop::Int, K::Int; verbose=false)
    output = Vector{Tuple{Float64, Int, Int}}()
    chunk = Chunk(copy(init_path), 1)
    range = start+1:stop-1

    for (iteration, i) in enumerate(range)
        verbose && (iteration % 100 == 0) && print("\33[2K [$K-NN] $(iteration)/$(length(range))\r")
        bv, bj = 0, 0

        for (_, j) in find_closest_cities(chunk.path, chunk.path[i], K)
            # Ignore neighbors behind the current position
            ((j == 1) || (j == length(chunk.path)) || (j < i)) && continue
            s = score_2opt(chunk, min(i, j), max(i, j))
            if s < bv
                bv, bj = s, j
            end
        end
        if bj != 0
            reverse!(chunk.path, min(i, bj), max(i, bj))
            s = score(chunk)
            reverse!(chunk.path, min(i, bj), max(i, bj))
            push!(output, (s, i, bj))
        end
    end

    output
end

start = parse(Int, ARGS[1])
stop = parse(Int, ARGS[2])
cities_fp = ARGS[3]
input_fp = ARGS[4]
output_fp = ARGS[5]
k = parse(Int, ARGS[6])

println("ga_2opt.jl start=$(start) stop=$(stop) cities_fp=$(cities_fp) input_fp=$(input_fp) output_fp=$(output_fp) k=$(k)")
println("Current dir is $(pwd())")

println("Loading cities...")
cities = read_cities(cities_fp)

println("Parsing path...")
path = map(s -> cities[parse(Int, s)+1], split(readline(input_fp)))

println("Starting 2-opt...")
output = nn_opt(path, start, stop, k)

output_arr = ["$(length(output))"]
for x in output
    push!(output_arr, "$(x[1]) $(x[2]) $(x[3])")
end

write(output_fp, join(output_arr, "\n"))