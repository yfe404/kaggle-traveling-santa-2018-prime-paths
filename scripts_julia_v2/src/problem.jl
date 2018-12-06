# Coord

struct Coord <: AbstractVector{Float64}
    x::Float64
    y::Float64
end

Base.eltype(::Type{Coord}) = Float64
Base.length(::Type{Coord}) = 2

Base.getindex(xy::Coord, i::Int) = i == 1 ? xy.x : xy.y
Base.size(::Coord) = (2,)

@fastmath distance(a::Coord, b::Coord) = sqrt((a.x-b.x)^2+(a.y-b.y)^2)

# City

struct City
    i::Int
    p::Bool
    xy::Coord
end

City(i::Int, xy::Coord) = City(i, isprime(i), xy)

City(i::Int, x::Float64, y::Float64) = City(i, isprime(i), Coord(x, y))

City(i::AbstractString, x::AbstractString, y::AbstractString) = City(parse(Int, i), parse(Float64, x), parse(Float64, y))

# Scoring

function score(path::Vector{City}; start=1)
    dist = 0.0
    @inbounds for i in 1:length(path)-1
        if ((i+(start-1)) % 10 == 0) && !path[i].p
            dist += distance(path[i].xy, path[i+1].xy)*1.1
        else
            dist += distance(path[i].xy, path[i+1].xy)
        end
    end
    dist
end

function verify!(path::Vector{City})
    @assert path[1].i == path[end].i == 0
    @assert length(unique(path)) == 197769
    true
end
