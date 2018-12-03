push!(LOAD_PATH, "src/")
import Pkg; Pkg.activate(".")
using Santa

# Score the swapping reversal of the path from k (inclusive) to l (inclusive)
# The lower the better
function score_2opt(chunk::Chunk, k::Int, l::Int)
    @assert 1 < k < l < length(chunk.path)

    # before: a k ... l b
    # after:  a l ... k b
    a, b = k-1, l+1
    
    p_a = is_penalized(chunk, a) ? 1.1 : 1.0
    p_l_b = is_penalized(chunk, l) ? 1.1 : 1.0 # Before
    p_k_b = !chunk.path[k].p && ((l+chunk.offset-1) % 10 == 0) ? 1.1 : 1.0 # After

    a_k = distance(chunk.path[a], chunk.path[k]) * p_a # Before
    a_l = distance(chunk.path[a], chunk.path[l]) * p_a # After

    l_b = distance(chunk.path[l], chunk.path[b]) * p_l_b # Before
    k_b = distance(chunk.path[k], chunk.path[b]) * p_k_b # After

    diff = (a_l - a_k) + (k_b - l_b)

    # Upper bound on max penalty gain for early exit ?
    
    penalties_diff = 0.0
    
    start = ((k+chunk.offset-1) % 10 == 0)*k
    if start == 0
        start = k+10-(k+chunk.offset-1) % 10
    end
    
    for i = start:10:l-1
        # Some distances are computed twice...
        penalties_diff -= !chunk.path[i].p * distance(chunk.path[i], chunk.path[i+1])
        penalties_diff += !chunk.path[l+k-i].p * distance(chunk.path[l+k-i], chunk.path[l+k-i-1])
    end

    penalties_diff *= 0.1
    diff + penalties_diff
end

function nn_opt(init_path::Vector{City}, start::Int, stop::Int, K::Int)
    chunk = Chunk(init_path[start:stop], start)
    path = copy(init_path) # We keep it to do scoring on full path
    output = Vector{Tuple{Float64, Int, Int}}()

    # NOTE: In this case we could precompute/cache closest cities since the path in chunk is not altered

    # Boundaries are left untouched
    for i = 2:length(chunk.path)-1 
        i % 100 == 0 && print("\33[2K [$K-NN] $(i)/$(length(chunk.path))\r")
        bv, bj = 0, 0
        for (_, j) in find_closest_cities(chunk.path, chunk.path[i], K)
            # Protect boundaries AND IGNORE NEIGHBORS BEHIND CURRENT POSITION (j < i)
            ((j == 1) || (j < i) || (j == length(chunk.path))) && continue
            # ------------------
            s = score_2opt(chunk, min(i,j), max(i,j))
            if s < bv
                bv, bj = s, j
            end
        end
        if bj != 0
            # Apply scoring on full path
            reverse!(init_path, min(i,bj), max(i,bj))
            s = score(init_path)
            reverse!(init_path, min(i,bj), max(i,bj))
            push!(output, (s, i+(start-1), bj+(start-1)))
        end
    end

    output
end

start = parse(Int, ARGS[1])
stop = parse(Int, ARGS[2])
cities_fp = ARGS[3]
input_fp = ARGS[4]
output_fp = ARGS[5]
k = parse(Int, ARGS[6])

println("ga_2opt.jl start=$(start) stop=$(stop) cities_fp=$(cities_fp) input_fp=$(input_fp) output_fp=$(output_fp) k=$(k)")
println("Current dir is $(pwd())")

println("Loading cities...")
cities = read_cities(cities_fp)

println("Parsing path...")
path = map(s -> cities[parse(Int, s)+1], split(readline(input_fp)))

println("Starting 2-opt...")
output = nn_opt(path, start, stop, k)

output_arr = ["$(length(output))"]
for x in output
    push!(output_arr, "$(x[1]) $(x[2]) $(x[3])")
end

write(output_fp, join(output_arr, "\n"))