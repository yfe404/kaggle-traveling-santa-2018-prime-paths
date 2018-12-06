function solve_greedy(cities::Vector{City}, K=1)
    cities_coords = map(c -> c.xy, cities)
    kdtree = KDTree(cities_coords)

    path = [cities[1]]
    path_idxs = Set([1])

    skip_fn(x::Int) = x in path_idxs

    for i = 1:length(cities)-1
        indices, _ = knn(kdtree, path[end].xy, min(length(cities)-i, K), false, skip_fn)
        index = rand(indices)
        push!(path, cities[index])
        push!(path_idxs, index)
    end
    
    push!(path, path[1])
    path
end

solve_random(cities::Vector{City}) = vcat(cities[1], shuffle(cities[2:end]), cities[1])
