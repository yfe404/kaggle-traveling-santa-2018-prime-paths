@everywhere using SharedArrays

@everywhere cities_fp = "../input/cities.csv"
@everywhere path_fp = "/home/maxmouchet/Downloads/1516917.csv"

@everywhere cities = read_cities(cities_fp)
@everywhere path = read_path(cities, path_fp)

# @everywhere path = path[1:250]
# @everywhere path[250] = path[1]

println("Original score: $(score(path))")

new_path = SharedArray{City}(length(path))
chunks = map(collect, get_chunks(nprocs()-1, length(path)))

# Protect boundaries
for chunk in chunks
    new_path[chunk[1]] = path[chunk[1]]
    new_path[chunk[end]] = path[chunk[end]]
    deleteat!(chunk, 1)
    deleteat!(chunk, length(chunk))
end

@sync @distributed for chunk in chunks
    println("$(chunk[1]):$(chunk[end])")
    new_path[chunk] = nn_opt(path[chunk], chunk[1], 1)
end

new_path = sdata(new_path)
new_score = score(new_path)
println(new_score)
verify!(new_path)

out = vcat("Path", map(c -> c.i, new_path))
write("sub_knn_opt_$(new_score).csv", join(out, "\n"))
