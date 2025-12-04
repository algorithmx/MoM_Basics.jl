using MoM_Basics
using Test
using StaticArrays

@testset "TriangleInfo IBC Extensions" begin
    # Test constructor initialization
    tri = TriangleInfo{Int64, Float64}(1)
    
    # Verify default Zs is zero
    @test tri.Zs == zero(Complex{Float64})
    @test tri.triID == 1
    
    # Verify we can set Zs
    test_Zs = 377.0 + 10.0im
    tri.Zs = test_Zs
    @test tri.Zs == test_Zs
    
    # Verify other fields still work as expected
    @test size(tri.vertices) == (3, 3)
    @test size(tri.inBfsID) == (3,)
end
