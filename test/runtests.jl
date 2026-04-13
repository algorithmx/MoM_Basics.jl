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

    @testset "Ports" begin
        include("ports.jl")
    end

    @testset "Port Masking" begin
        include("port_masking.jl")
    end

    @testset "Triangle IBC Extensions" begin
        include("triangles_ibc.jl")
    end

    @testset "Consolidated Extraction" begin
        include("consolidated_extraction.jl")
    end

    @testset "Triangle Connectivity" begin
        include("triangle_connectivity.jl")
    end

    @testset "Green's Functions" begin
        include("GreenFunction/runtests.jl")
    end
    
    rm("results"; force = true, recursive = true)

end
