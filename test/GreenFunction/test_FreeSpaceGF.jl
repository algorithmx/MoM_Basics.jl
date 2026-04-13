# test_FreeSpaceGF.jl
# Unit tests for FreeSpaceGF - Mathematical Correctness Verification
#
# Tests verify the equation: G(R) = exp(-jkR) / R
# From: layered_media_gf_research_materials.md Section 1.1

@testset "FreeSpaceGF Mathematical Correctness" begin

    @testset "Basic Evaluation: G(R) = exp(-jkR)/R" begin
        k = Complex(2π)  # λ = 1
        gf = FreeSpaceGF(k)
        
        r_src = [0.0, 0.0, 0.0]
        r_obs = [1.0, 0.0, 0.0]
        
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        R = 1.0
        expected = exp(-im*k*R) / R
        
        @test vals.g_A ≈ expected atol=1e-14
        @test vals.g_phi ≈ expected atol=1e-14
        @test vals.g_A == vals.g_phi  # Free-space identity
    end

    @testset "Mathematical Formula: G(R) = exp(-jkR)/R for various R and k" begin
        test_distances = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        test_wavenumbers = [1.0, 2π, 10.0, 100.0]
        
        for R in test_distances
            for k in test_wavenumbers
                gf = FreeSpaceGF(Complex(k))
                
                # Points separated by distance R
                r_src = [0.0, 0.0, 0.0]
                r_obs = [R, 0.0, 0.0]
                
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                expected = exp(-im*k*R) / R
                
                @test vals.g_A ≈ expected atol=1e-12
                @test vals.g_phi ≈ expected atol=1e-12
                @test vals.g_A == vals.g_phi
            end
        end
    end

    @testset "Reciprocity: G(r1, r2) = G(r2, r1)" begin
        k = Complex(2π)
        gf = FreeSpaceGF(k)
        
        # Random points
        test_points = [
            ([1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([-2.0, 3.0, -1.0], [4.0, -1.0, 2.0])
        ]
        
        for (r1, r2) in test_points
            vals_12 = evaluate_greenfunc(gf, r1, r2)
            vals_21 = evaluate_greenfunc(gf, r2, r1)
            
            @test vals_12.g_A ≈ vals_21.g_A atol=1e-14
            @test vals_12.g_phi ≈ vals_21.g_phi atol=1e-14
        end
    end

    @testset "Distance-Based Evaluation" begin
        k = Complex(2π)
        gf = FreeSpaceGF(k)
        
        test_R = [0.5, 1.0, 2.0, 5.0]
        
        for R in test_R
            # Using point-based evaluation
            r_src = [0.0, 0.0, 0.0]
            r_obs = [R, 0.0, 0.0]
            vals_point = evaluate_greenfunc(gf, r_obs, r_src)
            
            # Using distance-based evaluation
            vals_dist = evaluate_greenfunc(gf, R)
            
            @test vals_point.g_A ≈ vals_dist.g_A atol=1e-14
            @test vals_point.g_phi ≈ vals_dist.g_phi atol=1e-14
        end
    end

    @testset "Far-Field Decay: G(R→∞) → 0" begin
        k = Complex(2π)
        gf = FreeSpaceGF(k)
        
        # Very large distance
        R_large = 1e6
        r_src = [0.0, 0.0, 0.0]
        r_obs = [R_large, 0.0, 0.0]
        
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        
        # Should decay to near-zero (1/R decay)
        @test abs(vals.g_A) < 1e-5
        @test abs(vals.g_phi) < 1e-5
    end

    @testset "3D Distance: Non-axis-aligned points" begin
        k = Complex(2π)
        gf = FreeSpaceGF(k)
        
        r_src = [1.0, 2.0, 3.0]
        r_obs = [4.0, -1.0, 5.0]
        
        # Expected distance
        R = sqrt((4-1)^2 + (-1-2)^2 + (5-3)^2)
        expected = exp(-im*k*R) / R
        
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        
        @test vals.g_A ≈ expected atol=1e-14
        @test vals.g_phi ≈ expected atol=1e-14
    end

    @testset "Different Precisions" begin
        # Test with Float32
        k_f32 = Complex(Float32(2π))
        gf_f32 = FreeSpaceGF(k_f32)
        
        r_src = Float32[0.0, 0.0, 0.0]
        r_obs = Float32[1.0, 0.0, 0.0]
        
        vals = evaluate_greenfunc(gf_f32, r_obs, r_src)
        R = Float32(1.0)
        expected = exp(-im*k_f32*R) / R
        
        @test vals.g_A ≈ expected atol=1e-6
        @test vals.g_phi ≈ expected atol=1e-6
    end

    @testset "Display and Summary" begin
        k = Complex(2π)
        gf = FreeSpaceGF(k)
        
        # Test that display functions work without error
        @test sprint(show, gf) isa String
        @test sprint(summary, gf) isa String
    end

end
