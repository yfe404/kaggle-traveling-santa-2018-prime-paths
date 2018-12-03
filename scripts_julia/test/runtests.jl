import Pkg; Pkg.activate(".")
push!(LOAD_PATH, "src/");

using Test
using Santa

cities = read_cities("test/cities.csv")
path = read_path(cities, "test/1516917.csv")

@testset "Scoring" begin
    @test score(path) == 1.5169178997302188e6
    @test score(path) == score(path, start=1)
    @test score(path) == score(path[1:25]) + score(path[25:end], start=25)
    @test score(Chunk(path[467:end], 467)) == score(path[467:end], start=467)
end

@testset "2-opt" begin
    # Test with the full path
    c1 = Chunk(path, 1)

    # Edge cases are %10 +/- indices
    for i in [9,10,11], j in [19,20,21,34879]
        c2 = Chunk(reverse(path, i, j), 1)
        # Allow errors < 1e-6 since score_2opt may be more precise than
        # the score difference due to numerical errors
        @test score_2opt(c1, i, j) ≈ (score(c2) - score(c1)) atol=1e-6
    end

    # Test with a truncated path
    c1 = Chunk(path[1584:end], 1584)

    for i in [9,10,11], j in [19,20,21,34879]
        c2 = Chunk(reverse(path[1584:end], i, j), 1584)
        @test score_2opt(c1, i, j) ≈ (score(c2) - score(c1)) atol=1e-6
    end
end
