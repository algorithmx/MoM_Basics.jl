using MoM_Basics
using Test
using StaticArrays
using LinearAlgebra

@testset "Field Extraction" begin
    # Setup dummy data
    FT = Float64
    setPrecision!(FT)
    
    # Mock geometry
    # We need just enough structure to pass "hasproperty(geo, :center)"
    struct MockGeo
        center::SVec3D{FT}
    end
    
    geos = [MockGeo(SVec3D{FT}(0,0,0)), MockGeo(SVec3D{FT}(1,0,0))]
    
    # Mock source
    # PlaneWave(θ, ϕ, α, V)
    source = PlaneWave(π/2, 0.0, 0.0, 1.0) 
    
    # Initialize Parameters needed for field calc (e.g. K_0)
    # We can use inputBasicParameters to set defaults
    inputBasicParameters(frequency=1e8)
    
    # Test Calculation
    data = calExcitationFields(geos, source)
    
    @test data isa ExcitationFieldData
    @test data.npoints == 2
    @test data.positions[1] ≈ SVec3D{FT}(0,0,0)
    @test data.positions[2] ≈ SVec3D{FT}(1,0,0)
    
    # Check fields (E should be non-zero for plane wave)
    @test norm(data.E[1]) > 0
    @test norm(data.H[1]) > 0
    
    # Test Saving
    filename = "test_fields.csv"
    saveExcitationFields(filename, data)
    @test isfile(filename)
    
    # Verify content
    lines = readlines(filename)
    @test length(lines) == 3 # Header + 2 rows
    @test startswith(lines[1], "rx,ry,rz")
    
    # Cleanup
    rm(filename)
end
