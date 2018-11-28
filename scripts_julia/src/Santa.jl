__precompile__()

module Santa

import Primes: isprime

export
    City,
    distance,
    read_cities,
    read_path,
    score

struct City
    i::Int
    p::Bool
    x::Float64
    y::Float64
end

distance(a::City, b::City) = sqrt((a.x-b.x)^2+(a.y-b.y)^2)

function read_cities(fp::AbstractString)
    cities = Vector{City}()
    for line in readlines(fp)[2:end]
        i, x, y = map(x -> parse(Float64, x), split(line, ","))
        push!(cities, City(Int(i), isprime(Int(i)), x, y))
    end
    cities
end

function read_path(cities::Vector{City}, fp::AbstractString)
    lines = readlines(fp)[2:end]
    map(x -> cities[parse(Int, x)+1], lines)
end

function score(path::Vector{City})
    @assert (path[1].i == 0) && (path[end].i == 0)
    dist = 0.0
    @inbounds for i in 1:length(path)-1
        if (i % 10 == 0) && !path[i].p
            dist += distance(path[i], path[i+1])*1.1
        else
            dist += distance(path[i], path[i+1])
        end
    end
    dist
end

end