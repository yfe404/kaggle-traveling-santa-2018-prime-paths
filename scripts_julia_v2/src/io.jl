"""
    read_cities(fp::AbstractString)

Assume that the input file is sorted (city 0 is on the second line)
"""
read_cities(fp::AbstractString) = map(l -> City(split(l, ",")...), readlines(fp)[2:end])

"""
    read_path(cities::Vector{City}, fp::AbstractString)

Assume that `cities` is sorted (first city is city 0)
"""
read_path(cities::Vector{City}, fp::AbstractString) = map(x -> cities[parse(Int, x)+1], readlines(fp)[2:end])
