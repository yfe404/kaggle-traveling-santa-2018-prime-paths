__precompile__()

module Santa

import DataStructures: binary_maxheap, top
import Primes: isprime
import Random: shuffle

export
    # problem.jl
    City,
    distance,
    random_path,
    read_cities,
    read_path,
    score,
    verify!,
    Chunk,
    is_penalized,
    # tsplib.jl
    from_tsplib,
    to_tsplib,
    # knn.jl
    find_closest_cities,
    # opt.jl
    score_2opt

include("problem.jl")
include("tsplib.jl")
include("knn.jl")
include("opt.jl")

end
