using MoM_Basics, LinearAlgebra
using Test

@testset "MoM_Basics.jl" begin

    @testset "Params" begin
        include("params.jl")
    end

    @testset "Meshes" begin
        include("meshes.jl")
    end

    @testset "Basis functions" begin
        include("basis_functions.jl")
    end

    @testset "Sources" begin
        include("sources.jl")
    end

    @testset "Triangle IBC Extensions" begin
        include("triangles_ibc.jl")
    end

    @testset "Field Extraction" begin
        include("field_extraction.jl")
    end
    rm("results"; force = true, recursive = true)

end
