import Pkg; Pkg.activate(".")
push!(LOAD_PATH, "src/");

using Test
using Santa

cities = read_cities("test/cities.csv")
path = read_path(cities, "test/1516917.csv")

@testset "Scoring" begin
    @test score(path) == 1.5169178997302188e6
    @test score(path) == score(path, start=1)
end
