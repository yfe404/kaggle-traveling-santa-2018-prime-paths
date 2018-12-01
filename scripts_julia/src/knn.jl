# Fast k-nn lookup
function find_closest_cities(cities::Vector{City}, city::City, K::Int)
    K = min(K, length(cities))
    heap = binary_maxheap([(maxintfloat(Float64), 0)])
    for (i, c) in enumerate(cities)
        city.i == c.i && continue
        val = distance(city, c)
        if val < top(heap)[1]
            push!(heap, (val, i))
            if length(heap) > K
                pop!(heap)
            end
        end
    end
    heap.valtree
end
