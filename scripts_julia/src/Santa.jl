__precompile__()

module Santa

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
    to_tsplib

include("problem.jl")
include("tsplib.jl")

end
