using Test
using Random
using Santa

cities = read_cities("cities.csv")
path = read_path(cities, "1516917.csv")

Random.seed!(2018)

@testset "Scoring" begin
    @test score(path) == 1.5169178997302188e6
    @test score(path) == score(path, start=1)
    @test score(path) == score(path[1:25]) + score(path[25:end], start=25)
end

@testset "Solvers - Random" begin
    @test verify!(solve_random(cities))
end

@testset "Solvers - Greedy" begin
    @test verify!(solve_greedy(cities, 1))
    @test verify!(solve_greedy(cities, 10))
    @test score(solve_greedy(cities, 1)) == 1.8126021861388376e6
end

# @testset "2-opt" begin
#     # Test with the full path
#     c1 = Chunk(path, 1)

#     # Edge cases are %10 +/- indices
#     for i in [9,10,11], j in [19,20,21,34879]
#         c2 = Chunk(reverse(path, i, j), 1)
#         # Allow errors < 1e-6 since score_2opt may be more precise than
#         # the score difference due to numerical errors
#         @test score_2opt(c1, i, j) ≈ (score(c2) - score(c1)) atol=1e-6
#     end

#     # Test with a truncated path
#     c1 = Chunk(path[1584:end], 1584)

#     for i in [9,10,11], j in [19,20,21,34879]
#         c2 = Chunk(reverse(path[1584:end], i, j), 1584)
#         @test score_2opt(c1, i, j) ≈ (score(c2) - score(c1)) atol=1e-6
#     end
# end
