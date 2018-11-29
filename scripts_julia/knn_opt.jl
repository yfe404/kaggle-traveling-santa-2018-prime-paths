Base.eval(:(have_color = true))                                                                                      
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")

using Random
using Santa

import DataStructures: binary_maxheap, top

# Fast k-nn lookup
function find_closest_cities(cities::Vector{City}, city::City, K::Int)
    K = min(K, length(cities))
    heap = binary_maxheap([(maxintfloat(Float64), 0)])
    for (i, c) in enumerate(cities)
        city.i == c.i && continue
        val = distance(city, c)
        if val < top(heap)[1]
            push!(heap, (val, i))
            if length(heap) > K
                pop!(heap)
            end
        end
    end
    heap.valtree
end

function nn_opt(init_path::Vector{City}, indices::Vector{Int}, K::Int)
    path = copy(init_path)
    best = score(path)
    for (zzz, i) in enumerate(indices)
        zzz % 10 == 0 && print("\33[2K [$K-NN] $(zzz)/$(length(indices)) score = $(best)\r")
        bv, bj = best, 0
        for (_, j) in find_closest_cities(path, path[i], K)
            path[j].i == 0 && continue
            reverse!(path, min(i,j), max(i,j))
            s  = score(path)
            if s < bv
                bv, bj = s, j
            end
            reverse!(path, min(i,j), max(i,j))
        end
        if bj != 0
            reverse!(path,  min(i,bj), max(i,bj))
            best = bv
        end
    end
    path
end

println(Random.GLOBAL_RNG.seed)

cities = read_cities(ARGS[1])
path = read_path(cities, ARGS[2])

println("Original score: $(score(path))")

path = nn_opt(path, collect(10:10:length(path)-1), 100); println()
path = nn_opt(path, collect(length(path)-1:-1:2), 100); println()

out = vcat("Path", map(c -> c.i, path))
write("sub_knn_opt_$(score(path)).csv", join(out, "\n"))

