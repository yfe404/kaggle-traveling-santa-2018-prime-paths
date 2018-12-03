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