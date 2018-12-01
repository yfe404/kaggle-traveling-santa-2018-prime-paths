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
    # tsplib.jl
    from_tsplib,
    to_tsplib,
    # knn.jl
    find_closest_cities

include("problem.jl")
include("tsplib.jl")
include("knn.jl")

end
