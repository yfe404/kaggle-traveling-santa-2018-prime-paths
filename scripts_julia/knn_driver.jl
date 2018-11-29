@everywhere using SharedArrays

@everywhere cities_fp = "/home/ubuntu/cities.csv"
@everywhere path_fp = "/home/ubuntu/1516917.csv"
# @everywhere path_fp = "/home/ubuntu/1516773.csv"
# @everywhere path_fp = "sub_knn_opt_1.5167386354140872e6.csv"
# @everywhere path_fp = "sub_knn_opt_1.5167320695386857e6.csv"
# @everywhere path_fp = "sub_knn_opt_1.5167312153021502e6.csv"

@everywhere cities = read_cities(cities_fp)
@everywhere path = read_path(cities, path_fp)

# @everywhere path = path[1:250]
# @everywhere path[250] = path[1]

original_score = score(path)
println("Original score: $(original_score)")

new_path = SharedArray{City}(length(path))
chunks = map(collect, get_chunks(nprocs()-1, length(path)))

scores = [0, original_score]

while scores[end] != scores[end-1]
	@sync @distributed for chunk in chunks
    		println("$(chunk[1]):$(chunk[end])")
    		new_path[chunk] = nn_opt(path[chunk], chunk[1], 100)
	end

	@everywhere path = copy(sdata(new_path))
	push!(scores, score(path))
	println(scores[end-1])
	println(scores[end])
	verify!(path)

	out = vcat("Path", map(c -> c.i, path))
	write("sub_knn_opt_$(scores[end]).csv", join(out, "\n"))
	sleep(2)
end
