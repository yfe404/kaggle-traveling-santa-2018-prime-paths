Base.eval(:(have_color = true))
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")
import DataStructures: binary_maxheap, top

using Santa

function nn_opt(init_path::Vector{City}, start::Int, K::Int)
    path = copy(init_path)
    best = score(path, start=start)
    println(best)
    for i = 2:length(path)-1 # Boundaries are left untouched
        i % 1000 == 0 && println("$(i)/$(length(path))")
        bv, bj = best, 0
        for (_, j) in find_closest_cities(path, path[i], K)
            ((j == 1) || (j == length(path))) && continue # Boundaries are left untouched
            reverse!(path, min(i,j), max(i,j))
            s  = score(path, start=start)
            if s < bv
                bv, bj = s, j
            end
            reverse!(path, min(i,j), max(i,j))
        end
        if bj != 0
            reverse!(path,  min(i,bj), max(i,bj))
            best = bv
            println(best)
        end
    end
    path
end

# Returns k chunks of a vector of size n
function get_chunks(k::Int, n::Int)
    idxs = collect(1:ceil(Int,n/k):n)
    push!(idxs, n+1)
    chunks = []
    for i = 1:length(idxs)-1
        push!(chunks, idxs[i]:idxs[i+1]-1)
    end
    chunks
end

