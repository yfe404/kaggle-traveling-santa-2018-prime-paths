function modrange(start, stop, n)
    @assert stop >= start >= 0
    if start % n == 0
        return start:n:stop
    else
        return (start+n-(start%n)):n:stop
    end
end

# 2-opt

function score_2opt(path::Vector{City}, k::Int, l::Int)
    @assert 1 < k < l < length(path)

    # before: a k ... l b
    # after:  a l ... k b
    a, b = k-1, l+1

    p_a   = !path[a].p && (a % 10 == 0) ? 1.1 : 1.0 # Before
    p_l_b = !path[l].p && (l % 10 == 0) ? 1.1 : 1.0 # Before
    p_k_b = !path[k].p && (l % 10 == 0) ? 1.1 : 1.0 # After

    a_k = distance(path[a].xy, path[k].xy) * p_a # Before
    a_l = distance(path[a].xy, path[l].xy) * p_a # After

    l_b = distance(path[l].xy, path[b].xy) * p_l_b # Before
    k_b = distance(path[k].xy, path[b].xy) * p_k_b # After

    diff = (a_l - a_k) + (k_b - l_b)

    # Upper bound on max penalty gain for early exit ?

    penalties_diff = 0.0

    for i in modrange(k, l-1, 10)
        penalties_diff +=
            (!path[l+k-i].p * distance(path[l+k-i].xy, path[l+k-i-1].xy)) -
            (!path[i].p * distance(path[i].xy, path[i+1].xy))
    end

    diff + penalties_diff*0.1
end

# 3-opt

# TODO