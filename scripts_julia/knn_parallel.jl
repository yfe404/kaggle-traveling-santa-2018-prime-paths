Base.eval(:(have_color = true))
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")
import DataStructures: binary_maxheap, top

using Santa

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

function score_partial(path::Vector{City}, start::Int)
    dist = 0.0
    @inbounds for i in 1:length(path)-1
        if ((i+(start-1)) % 10 == 0) && !path[i].p
            dist += distance(path[i], path[i+1])*1.1
        else
            dist += distance(path[i], path[i+1])
        end
    end
    dist
end

function nn_opt(init_path::Vector{City}, start::Int, K::Int)
    path = copy(init_path)
    best = score_partial(path, start)
    println(best)
    for i = 2:length(path)-1 # Boundaries are left untouched
        i % 1000 == 0 && println("$(i)/$(length(path))")
        bv, bj = best, 0
        for (_, j) in find_closest_cities(path, path[i], K)
            ((j == 1) || (j == length(path))) && continue # Boundaries are left untouched
            reverse!(path, min(i,j), max(i,j))
            s  = score_partial(path, start)
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

