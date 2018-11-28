format_tsplib(x::Float64) = round(Int, x*1000)

function to_tsplib(fp::AbstractString, path::Vector{City})
    out = [
        "NAME : traveling-santa-2018-prime-paths",
        "TYPE : TSP",
        "DIMENSION : $(length(path))",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION"
    ]
    for (i, city) in enumerate(path)
        push!(out, "$i $(format_tsplib(city.x)) $(format_tsplib(city.y))")
    end
    push!(out, "EOF")
    res = join(out, "\n")
    write(fp, res)
end

function from_tsplib(fp::AbstractString, cities::Vector{City})
    coords_to_cities = Dict([("$(format_tsplib(city.x)),$(format_tsplib(city.y))" => city) for city in cities])
    path = Vector{City}()
    for line in readlines(fp)
        # TODO: Cleanup
        (occursin(":", line) || (line == "NODE_COORD_SECTION") || (line == "EOF")) && continue
        i, x, y = split(line)
        push!(path, coords_to_cities["$(round(Int, parse(Float64, x))),$(round(Int, parse(Float64, y)))"])
    end
    path
end
