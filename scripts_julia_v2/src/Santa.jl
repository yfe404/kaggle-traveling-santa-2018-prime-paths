__precompile__()

module Santa

import NearestNeighbors: KDTree, knn
import Primes: isprime
import Random: shuffle

export
    # problem.jl
    City,
    Coord,
    distance,
    score,
    verify!,
    # io.jl
    read_cities,
    read_path,
    # solver.jl
    solve_greedy,
    solve_random

include("problem.jl")
include("io.jl")
include("solvers.jl")

end