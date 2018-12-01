struct City
    i::Int
    p::Bool
    x::Float64
    y::Float64
end

City(i::Int, x::Float64, y::Float64) = City(i, isprime(i), x, y)

City(i::AbstractString, x::AbstractString, y::AbstractString) = City(parse(Int, i), parse(Float64, x), parse(Float64, y))

# /!\ Assume that the input file is sorted (city 0 is on the second line)
read_cities(fp::AbstractString) = map(l -> City(split(l, ",")...), readlines(fp)[2:end])

# /!\ Assume that `cities` is sorted (first city is city 0)
read_path(cities::Vector{City}, fp::AbstractString) = map(x -> cities[parse(Int, x)+1], readlines(fp)[2:end])

@fastmath distance(a::City, b::City) = sqrt((a.x-b.x)^2+(a.y-b.y)^2)

@views random_path(cities::Vector{City}) = vcat(cities[1], shuffle(cities[2:end]), cities[1])

function score(path::Vector{City}; start=1)
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

function verify!(path::Vector{City})
    @assert path[1].i == path[end].i == 0
    @assert length(unique(path)) == 197769
end
