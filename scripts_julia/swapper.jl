Base.eval(:(have_color = true))                                                                                      
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")

using ProgressMeter
using Santa

function swap!(v::Vector{T}, a::Int, b::Int) where T
    v[a], v[b] = v[b], v[a]
end

function hill_climbing(init_path::Vector{City}, iterations::Int)
    best = score(init_path)
    path = copy(init_path)
    r = 2:length(init_path)-1
    @showprogress for i = 1:iterations
        a, b = rand(r, 2)
        swap!(path, a, b)
        _score = score(path)
        if _score < best
            best = _score
            println(best)
        else
            swap!(path, a, b)
        end
    end
    path
end

cities = read_cities(ARGS[1])
path = read_path(cities, ARGS[2])

# TODO: Print seed

hill_climbing(path, parse(Int, ARGS[3]))

