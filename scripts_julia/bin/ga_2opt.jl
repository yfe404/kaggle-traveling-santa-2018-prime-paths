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

    # Original full path (from city 0 to city 0) for computing the end-to-end score
    path = copy(init_path)

    # Subset of the full path on which we search for the optimal 2-opt for every cities
    chunk = Chunk(init_path[start:stop], start)

    # Boundaries are left untouched
    for chunk_i = 2:length(chunk.path)-1 
        verbose && (chunk_i % 100 == 0) && print("\33[2K [$K-NN] $(i)/$(length(chunk.path))\r")

        # best score diff, index of the best target swap in the full path ! (not chunk_j)
        bv, bj = 0, 0

        # NOTE: We perform the neighbor search on the full path !
        for (_, path_j) in find_closest_cities(path, chunk.path[chunk_i], K)
            chunk_j = path_j-start+1

            # Protect boundaries AND IGNORE NEIGHBORS BEHIND CURRENT POSITION (j < i)
            ((chunk_j <= 1) || (chunk_j < chunk_i) || (chunk_j == length(chunk.path))) && continue

            s = score_2opt(chunk, min(chunk_i, chunk_j), max(chunk_i, chunk_j))
            if s < bv
                bv, bj = s, path_j
            end
        end
        if bj != 0
            # Apply scoring on full path
            path_i = chunk_i+start-1
            reverse!(path, min(path_i, bj), max(path_i, bj))
            s = score(path)
            reverse!(path, min(path_i, bj), max(path_i, bj))
            push!(output, (s, path_i, bj))
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