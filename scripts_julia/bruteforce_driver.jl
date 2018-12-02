# TODO: Use DArrays instead
@everywhere using SharedArrays

@everywhere cities_fp = "test/cities.csv"
# @everywhere path_fp = "outputs/1516917.csv"
@everywhere path_fp = "sub_perm_opt_1.5166946654302643e6.csv"

@everywhere cities = read_cities(cities_fp)
@everywhere path = read_path(cities, path_fp)

original_score = score(path)
println("Original score: $(original_score)")

new_path = SharedArray{City}(length(path))
chunks = map(collect, get_chunks(nprocs()-1, length(path)))

scores = [0, original_score]

while scores[end] != scores[end-1]
	@sync @distributed for chunk in chunks
		println("$(chunk[1]):$(chunk[end])")
		new_path[chunk] = perm_opt(Chunk(path[chunk], chunk[1]), 3).path
	end

	@everywhere path = copy(sdata(new_path))
	push!(scores, score(path))
	println(scores[end-1])
	println(scores[end])
	verify!(path)

	out = vcat("Path", map(c -> c.i, path))
	write("sub_perm_opt_$(scores[end]).csv", join(out, "\n"))
	sleep(2)
end
