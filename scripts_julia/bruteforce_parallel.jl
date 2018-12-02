Base.eval(:(have_color = true))
push!(LOAD_PATH, "src/")

import Pkg; Pkg.activate(".")
import Combinatorics: permutations

using Santa

# TODO: OPTIMIZE
function perm_opt(init_chunk::Chunk, Δ::Int)
    chunk = Chunk(copy(init_chunk.path), copy(init_chunk.offset))

    for i = Δ+2:length(chunk.path)-(Δ+2)
        i % 100 == 0 && println("$(i)/$(length(chunk.path))")
        best, perm = score(chunk.path[i-Δ-1:i+Δ+1], start=(chunk.offset-1)+(i-Δ-1)), []
        for j in permutations(i-Δ:i+Δ)
            s = score(vcat(chunk.path[i-Δ-1], chunk.path[j], chunk.path[i+Δ+1]), start=(chunk.offset-1)+(i-Δ-1))
            if s < best
                best, perm = s, j
            end
        end
        if perm != []
            chunk.path[i-Δ:i+Δ] = chunk.path[perm]
            println(score(chunk))
        end
    end

    chunk
end

# Returns k chunks of a vector of size n
function get_chunks(k::Int, n::Int)
    idxs = collect(1:ceil(Int,n/k):n)
    push!(idxs, n+1)
    chunks = []
    for i = 1:length(idxs)-1
        push!(chunks, idxs[i]:idxs[i+1]-1)
    end
    chunks
end
