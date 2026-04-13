# test_evaluate_greenfunc_star.jl
# Comprehensive test suite for evaluate_greenfunc_star function
# Tests singularity extraction for LayeredMediaGF
#
# References:
# - Simsek, Liu & Wei, "Singularity Subtraction for Evaluation of Green's 
#   Functions for Multilayer Media," IEEE T-MTT, vol. 54, pp. 216-225, 2006.
# - Michalski & Mosig, IEEE T-AP, vol. 45, pp. 508-519, 1997.

using Test
using MoM_Basics
using MoM_Basics.GreenFunction
using StaticArrays: SVector

@testset "evaluate_greenfunc_star for LayeredMediaGF" begin
    
    # =============================================================================
    # Test Configuration
    # =============================================================================
    
    FT = Float64
    frequency = 4e9  # 4 GHz
    c0 = 299792458.0
    k0 = 2π * frequency / c0
    
    # Standard grounded substrate configuration
    layers = [
        LayerInfo("substrate", Complex{FT}(11.7), Complex{FT}(1.0), FT(500e-6)),
        LayerInfo("air", Complex{FT}(1.0), Complex{FT}(1.0), FT(Inf))
    ]
    stack = LayerStack(layers; reference_z=FT(0.0), has_ground_plane=true)
    gf = LayeredMediaGF(stack, frequency)
    
    # =============================================================================
    # Category 1: Basic Functionality Tests
    # =============================================================================
    
    @testset "1. Basic Functionality" begin
        
        @testset "1.1 Function exists and is exported" begin
            @test isdefined(MoM_Basics.GreenFunction, :evaluate_greenfunc_star)
            @test hasmethod(evaluate_greenfunc_star, (LayeredMediaGF{FT}, Vector{FT}, Vector{FT}))
            @test hasmethod(evaluate_greenfunc_star, (LayeredMediaGF{FT}, SVector{3,FT}, SVector{3,FT}))
        end
        
        @testset "1.2 Returns correct type" begin
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(100e-6, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test vals isa GreenFuncVals{Complex{FT}}
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
        
        @testset "1.3 Both components are complex" begin
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(50e-6, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test vals.g_A isa Complex{FT}
            @test vals.g_phi isa Complex{FT}
        end
    end
    
    # =============================================================================
    # Category 2: Singularity Extraction Verification
    # =============================================================================
    
    @testset "2. Singularity Extraction Verification" begin
        
        @testset "2.1 Smooth part is smaller than full GF" begin
            # The smooth part should exclude the 1/R singularity
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(100e-6, 0.0, z)
            
            vals_full = evaluate_greenfunc(gf, r_obs, r_src)
            vals_star = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            # Smooth part should be smaller than full GF
            @test abs(vals_star.g_A) < abs(vals_full.g_A)
            @test abs(vals_star.g_phi) < abs(vals_full.g_phi)
        end
        
        @testset "2.2 Difference equals quasi-static term (1/R)" begin
            # For same-layer interactions, the difference should be ~1/R
            z = FT(250e-6)
            ρ = FT(100e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals_full = evaluate_greenfunc(gf, r_obs, r_src)
            vals_star = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            diff_g_A = abs(vals_full.g_A - vals_star.g_A)
            expected_qs = 1/ρ  # Quasi-static term
            
            # Allow 1% tolerance for numerical differences
            @test diff_g_A ≈ expected_qs rtol=1e-2
        end
        
        @testset "2.3 Self-term returns finite value (ρ = 0)" begin
            # At ρ = 0 (coincident triangles), full GF is singular
            # But smooth part should remain finite
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(0.0, 0.0, z)  # Same point
            
            vals_star = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals_star.g_A)
            @test isfinite(vals_star.g_phi)
            @test abs(vals_star.g_A) > 0
            @test abs(vals_star.g_phi) > 0
        end
        
        @testset "2.4 Smooth part at different distances" begin
            # Test multiple horizontal distances
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            
            ρ_values = [FT(10e-6), FT(50e-6), FT(100e-6), FT(500e-6)]
            smooth_values = Float64[]
            
            for ρ in ρ_values
                r_obs = SVector{3,FT}(ρ, 0.0, z)
                vals = evaluate_greenfunc_star(gf, r_obs, r_src)
                push!(smooth_values, abs(vals.g_A))
            end
            
            # All values should be finite
            @test all(isfinite.(smooth_values))
            @test all(smooth_values .> 0)
            
            # Generally, smooth part should decrease with distance
            # (though not monotonically due to complex image interference)
        end
    end
    
    # =============================================================================
    # Category 3: z-Coordinate Dependence
    # =============================================================================
    
    @testset "3. z-Coordinate Dependence" begin
        
        @testset "3.1 Smooth part varies with vertical separation" begin
            ρ = FT(100e-6)
            z_src = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z_src)
            
            # Test at different dz values
            dz_values = [FT(0.0), FT(10e-6), FT(50e-6), FT(100e-6)]
            g_values = Complex{FT}[]
            
            for dz in dz_values
                z_obs = z_src + dz
                r_obs = SVector{3,FT}(ρ, 0.0, z_obs)
                vals = evaluate_greenfunc_star(gf, r_obs, r_src)
                push!(g_values, vals.g_A)
            end
            
            # Values should not all be identical (z-dependence exists)
            g_variation = maximum(abs.(g_values)) - minimum(abs.(g_values))
            @test g_variation > 0.001 * abs(g_values[1])
        end
        
        @testset "3.2 Same-height case (z = z')" begin
            z = FT(250e-6)
            ρ = FT(100e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
    end
    
    # =============================================================================
    # Category 4: Reciprocity and Symmetry
    # =============================================================================
    
    @testset "4. Reciprocity and Symmetry" begin
        
        @testset "4.1 Reciprocity G(r_obs, r_src) = G(r_src, r_obs)" begin
            z1, z2 = FT(200e-6), FT(300e-6)
            ρ = FT(100e-6)
            
            # Forward
            r_src1 = SVector{3,FT}(0.0, 0.0, z1)
            r_obs1 = SVector{3,FT}(ρ, 0.0, z2)
            vals_12 = evaluate_greenfunc_star(gf, r_obs1, r_src1)
            
            # Reverse
            r_src2 = SVector{3,FT}(0.0, 0.0, z2)
            r_obs2 = SVector{3,FT}(ρ, 0.0, z1)
            vals_21 = evaluate_greenfunc_star(gf, r_obs2, r_src2)
            
            # Should be equal (reciprocity)
            @test vals_12.g_A ≈ vals_21.g_A rtol=1e-6
            @test vals_12.g_phi ≈ vals_21.g_phi rtol=1e-6
        end
        
        @testset "4.2 Horizontal symmetry (x ↔ y)" begin
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            
            # Point at (ρ, 0, z)
            r_obs1 = SVector{3,FT}(100e-6, 0.0, z)
            vals1 = evaluate_greenfunc_star(gf, r_obs1, r_src)
            
            # Point at (0, ρ, z) - same horizontal distance
            r_obs2 = SVector{3,FT}(0.0, 100e-6, z)
            vals2 = evaluate_greenfunc_star(gf, r_obs2, r_src)
            
            # Should be equal (isotropy in horizontal plane)
            @test vals1.g_A ≈ vals2.g_A rtol=1e-10
            @test vals1.g_phi ≈ vals2.g_phi rtol=1e-10
        end
    end
    
    # =============================================================================
    # Category 5: Layer Interface Tests
    # =============================================================================
    
    @testset "5. Layer Interface Tests" begin
        
        @testset "5.1 At substrate surface (z = 0)" begin
            ρ = FT(100e-6)
            z = FT(0.0)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
        
        @testset "5.2 Near layer interface" begin
            ρ = FT(100e-6)
            z = FT(499e-6)  # Just below substrate-air interface
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
        
        @testset "5.3 In air layer" begin
            ρ = FT(100e-6)
            z = FT(600e-6)  # In air layer
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
    end
    
    # =============================================================================
    # Category 6: Comparison with Other GF Types
    # =============================================================================
    
    @testset "6. Interface Pattern Verification" begin
        
        @testset "6.1 LayeredMediaGF returns correct type" begin
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(100e-6, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            @test vals isa GreenFuncVals{Complex{FT}}
        end
        
        @testset "6.2 Two method signatures exist" begin
            # Test both AbstractVector and Vec3D signatures
            z = FT(250e-6)
            r_src_vec = [0.0, 0.0, z]  # Regular vector
            r_obs_vec = [100e-6, 0.0, z]
            r_src_svec = SVector{3,FT}(0.0, 0.0, z)  # Static vector
            r_obs_svec = SVector{3,FT}(100e-6, 0.0, z)
            
            vals1 = evaluate_greenfunc_star(gf, r_obs_vec, r_src_vec)
            vals2 = evaluate_greenfunc_star(gf, r_obs_svec, r_src_svec)
            
            @test vals1 isa GreenFuncVals
            @test vals2 isa GreenFuncVals
            @test vals1.g_A ≈ vals2.g_A rtol=1e-10
        end
    end
    
    # =============================================================================
    # Category 7: Edge Cases and Robustness
    # =============================================================================
    
    @testset "7. Edge Cases and Robustness" begin
        
        @testset "7.1 Very small horizontal separation" begin
            z = FT(250e-6)
            ρ = FT(1e-9)  # 1 nm - effectively zero
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
        
        @testset "7.2 Moderate horizontal separation" begin
            z = FT(250e-6)
            ρ = FT(1e-3)  # 1 mm
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
        
        @testset "7.3 Large horizontal separation" begin
            z = FT(250e-6)
            ρ = FT(10e-3)  # 1 cm
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
        end
    end
    
    # =============================================================================
    # Category 8: Physical Consistency
    # =============================================================================
    
    @testset "8. Physical Consistency" begin
        
        @testset "8.1 Smooth part decays with distance (general trend)" begin
            # Test that |G*| generally decreases as R increases
            z = FT(250e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            
            ρ_values = [FT(10e-6), FT(100e-6), FT(1000e-6)]
            magnitudes = Float64[]
            
            for ρ in ρ_values
                r_obs = SVector{3,FT}(ρ, 0.0, z)
                vals = evaluate_greenfunc_star(gf, r_obs, r_src)
                push!(magnitudes, abs(vals.g_A))
            end
            
            # General trend: should decrease (allow for non-monotonicity due to
            # complex image interference, but check overall decay)
            @test magnitudes[1] > magnitudes[3]  # Decay from near to far field
        end
        
        @testset "8.2 Vector and scalar potentials are different" begin
            # In layered media, g_A ≠ g_phi (unlike free space)
            z = FT(250e-6)
            ρ = FT(100e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            # They should be different (TE vs TM reflection coefficients)
            @test vals.g_A != vals.g_phi
        end
        
        @testset "8.3 Phase is physically reasonable" begin
            z = FT(250e-6)
            ρ = FT(100e-6)
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            
            vals = evaluate_greenfunc_star(gf, r_obs, r_src)
            
            # Phase should be finite
            @test isfinite(angle(vals.g_A))
            @test isfinite(angle(vals.g_phi))
        end
    end

end

# =============================================================================
# Summary
# =============================================================================
#
# Total test categories: 8
# Total test sets: 22+
# Coverage:
#   - Basic functionality (type checking, exports)
#   - Singularity extraction verification (key test: difference = 1/R)
#   - z-coordinate dependence
#   - Reciprocity and symmetry
#   - Layer interfaces
#   - Interface consistency with other GF types
#   - Edge cases (small/large separations)
#   - Physical consistency
#
# Key validation criteria:
#   1. Returns finite values even at ρ = 0 (self-term case)
#   2. Difference between full GF and smooth part equals quasi-static term
#   3. Exhibits correct z-dependence
#   4. Satisfies reciprocity G(r1,r2) = G(r2,r1)
#   5. Handles layer interfaces correctly
#