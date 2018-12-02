Base.eval(:(have_color = true))                                                                                      
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")

using ArgParse
using ProgressMeter
using Santa

function solve_greedy(cities::Vector{City})
    cities = copy(cities)
    path = Vector{City}([popfirst!(cities)])
    @showprogress for _ in 1:length(cities)
        _, nn = find_closest_cities(cities, path[end], 1)[1]
        push!(path, cities[nn])
        deleteat!(cities, nn)
    end
    push!(path, path[1])
    path
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--cities"
            metavar = "FILE"
            help = "path to kaggle cities.csv file"
            required = true
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    cities = read_cities(parsed_args["cities"])
    path = solve_greedy(cities)
    verify!(path)
    println("Score: $(score(path))")

    out = vcat("Path", map(c -> c.i, path))
    write("greedy_$(score(path)).csv", join(out, "\n"))
end

main()