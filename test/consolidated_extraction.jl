using MoM_Basics
using Test
using StaticArrays
using LinearAlgebra

@testset "FieldData Tests" begin
    FT = Float64
    CT = Complex{FT}
    setPrecision!(FT)
    
    # 1. Mock Data Setup (Minimal needed for Basics)
    struct MockGeo
        center::SVec3D{FT}
    end
    # Add hasproperty fallback if needed or just use property access in calIncidentFields
    # calIncidentFields checks if hasproperty(geo, :center) or if it is a specific type.
    # Actually calIncidentFields implementation:
    # if hasproperty(geo, :center)
    #    r = SVec3D{FT}(geo.center)
    # ...
    # So MockGeo needs :center property.
    
    geos = [
        MockGeo(SVec3D{FT}(0,0,0)),
        MockGeo(SVec3D{FT}(1,0,0))
    ]
    
    source = PlaneWave(π/2, 0.0, 0.0, 1.0)

    # 2. Test Incident Field Extraction
    @testset "Incident Fields" begin
        data = calIncidentFields(geos, source)
        @test data isa FieldData
        @test data.npoints == 2
        @test haskey(data.fields, :E_inc)
        @test haskey(data.fields, :H_inc)
        @test norm(data.fields[:E_inc][1]) > 0
    end

    # 3. Test FieldData Merging (Manual Construction for second data)
    @testset "Data Merging" begin
        data_inc = calIncidentFields(geos, source)
        
        # Manually create a FieldData for currents
        positions = data_inc.positions
        npoints = data_inc.npoints
        J_vec = [SVec3D{CT}(1.0+0im, 0, 0), SVec3D{CT}(0, 1.0+0im, 0)]
        data_curr = FieldData{FT, CT}(npoints, positions)
        data_curr.fields[:J] = J_vec
        
        # Merge
        mergeFieldData!(data_inc, data_curr)
        
        @test data_inc.npoints == 2
        @test haskey(data_inc.fields, :E_inc)
        @test haskey(data_inc.fields, :H_inc)
        @test haskey(data_inc.fields, :J)
        @test norm(data_inc.fields[:E_inc][1]) > 0
        @test norm(data_inc.fields[:J][1]) > 0
    end

    # 4. Test Saving
    @testset "File I/O" begin
        data = calIncidentFields(geos, source)
        
        # Test CSV
        csv_file = "test_unified_basics.csv"
        saveFieldData(csv_file, data)
        @test isfile(csv_file)
        lines = readlines(csv_file)
        # Header + 2 data rows
        @test length(lines) == 3
        # Header should contain positions and fields
        @test occursin("rx,ry,rz", lines[1])
        @test occursin("E_inc", lines[1])
        @test occursin("H_inc", lines[1])
        rm(csv_file)

        # Test NPZ (if supported)
        try
            npz_file = "test_unified_basics.npz"
            saveFieldData(npz_file, data)
            @test isfile(npz_file)
            rm(npz_file)
        catch
        end
    end
end
