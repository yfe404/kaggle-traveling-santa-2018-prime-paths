Base.eval(:(have_color = true))                                                                                      
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")

using ArgParse
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

function nn_opt(init_chunk::Chunk, K::Int)
    chunk = Chunk(copy(init_chunk.path), copy(init_chunk.offset))
    best = score(chunk.path)

    # Boundaries are left untouched
    for i = 2:length(chunk.path)-1 
        i % 100 == 0 && print("\33[2K [$K-NN] $(i)/$(length(chunk.path)) score = $(best)\r")
        bv, bj = 0, 0
        for (_, j) in find_closest_cities(chunk.path, chunk.path[i], K)
            # Protect boundaries
            ((j == 1) || (j == length(chunk.path))) && continue
            # ------------------
            s = score_2opt(chunk, min(i,j), max(i,j))
            if s < bv
                bv, bj = s, j
            end
        end
        if bj != 0
            reverse!(chunk.path,  min(i,bj), max(i,bj))
            # println(bv)
            best = score(chunk.path)
        end
    end

    chunk
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--cities"
            metavar = "FILE"
            help = "path to kaggle cities.csv file"
            required = true
        "--path"
            metavar = "FILE"
            help = "path to the file containing the solution to optimize, in kaggle .csv submission format"
            required = true
        "--knn"
            metavar = "K"
            help = "number of nearest neighbors to consider in the 2-opt"
            required = true
            arg_type = Int
        "--loop"
            action = :store_true
            help = "loop until no further improvement are made"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    cities = read_cities(parsed_args["cities"])
    path = read_path(cities, parsed_args["path"])

    verify!(path)
    println("Original score: $(score(path))")

    chunk = Chunk(path, 1)
    last_score = score(chunk)

    while true
        chunk = nn_opt(chunk, parsed_args["knn"])
        new_score = score(chunk)
        println("")

        if (new_score < last_score) && parsed_args["loop"]
            last_score = new_score
        else
            break
        end
    end

    out = vcat("Path", map(c -> c.i, chunk.path))
    write("sub_knn_opt_$(score(chunk.path)).csv", join(out, "\n"))
end

main()