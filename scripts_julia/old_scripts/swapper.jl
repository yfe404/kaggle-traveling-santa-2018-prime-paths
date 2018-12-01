Base.eval(:(have_color = true))                                                                                      
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")

using Random
using Santa

function swap!(v::Vector{T}, a::Int, b::Int) where T
    v[a], v[b] = v[b], v[a]
end

function hill_climbing(init_path::Vector{City}, iterations::Int)
    best = score(init_path)
    path = copy(init_path)
    r = 2:length(init_path)-1

    t, speed = time(), 0

    for i = 1:iterations
        if i % 100 == 0
            speed = round(100 / (time() - t))
            print("\33[2K [$(speed) iterations/s] $(i)/$(iterations) score = $(best)\r")
            t = time()
        end

        a, b = rand(r, 2)
        swap!(path, a, b)
        _score = score(path)
        if _score < best
            best = _score
        else
            swap!(path, a, b)
        end
    end
    path
end

println(Random.GLOBAL_RNG.seed)

cities = read_cities(ARGS[1])
path = read_path(cities, ARGS[2])

# path = copy(cities[2:end])
# shuffle!(path)

path = filter(x -> x.i > 0, path)
shuffle!(path)

pushfirst!(path, cities[1])
push!(path, cities[1])

hill_climbing(path, parse(Int, ARGS[3]))
