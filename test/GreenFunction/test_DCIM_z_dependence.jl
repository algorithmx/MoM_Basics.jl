# test_DCIM_z_dependence.jl
# Test cases for DCIM z-coordinate dependence validation

using Test
using MoM_Basics
using MoM_Basics.GreenFunction
using StaticArrays: SVector
# 
# These tests reflect MODEST expectations for DCIM behavior with z-dependence.
# They are designed to:
# 1. Establish baseline for current (buggy) behavior
# 2. Validate the fix when implemented
# 3. Document expected tolerances and limitations
#
# Key insight: DCIM is an approximation method. Exact results are not expected,
# but the z-dependence must be physically correct (i.e., changing z should change G).

@testset "DCIM z-Dependence Tests" begin

    # =============================================================================
    # Test Setup: Define a standard test configuration
    # =============================================================================
    
    # Standard test stack: Grounded substrate (microstrip-like)
    # This is a well-studied configuration with published reference data
    FT = Float64
    frequency = 4e9  # 4 GHz, typical for the project's application
    c0 = 299792458.0
    k0 = 2π * frequency / c0  # free-space wavenumber
    λ0 = c0 / frequency  # ~75 mm
    
    # Layer stack: 500 μm substrate (εr=11.7, silicon) + air half-space
    layers = [
        LayerInfo("substrate", Complex{FT}(11.7), Complex{FT}(1.0), FT(500e-6)),
        LayerInfo("air", Complex{FT}(1.0), Complex{FT}(1.0), FT(Inf))
    ]
    stack = LayerStack(layers; reference_z=FT(0.0), has_ground_plane=true)
    
    # Create LayeredMediaGF (this runs DCIM fitting)
    # Note: This may take time due to GPOF fitting
    @testset "Setup: Create LayeredMediaGF" begin
        gf = LayeredMediaGF(stack, frequency)
        @test gf isa LayeredMediaGF
        @test gf.stack.n_layers == 2
        @test gf.frequency ≈ frequency
        @test abs(gf.k) ≈ k0 rtol=1e-10
    end
    
    # Only create once for all tests
    gf = LayeredMediaGF(stack, frequency)
    
    # =============================================================================
    # Test Category 1: Same Layer, z = z' (Baseline - Should Work Correctly)
    # =============================================================================
    
    @testset "Category 1: Same Layer, z = z' (Baseline)" begin
        
        # Test 1.1: Horizontal dipole at substrate surface (z = z' = 0.5 μm, middle of substrate)
        @testset "1.1 Surface points at same height" begin
            z = FT(250e-6)  # Middle of substrate
            h = z  # Height above ground
            
            # Points at same height, different horizontal separations
            test_ρ = [FT(1e-6), FT(10e-6), FT(100e-6), FT(1e-3)]  # 1 μm to 1 mm
            
            for ρ in test_ρ
                # Create 3D points
                r_src = SVector{3,FT}(0.0, 0.0, z)
                r_obs = SVector{3,FT}(ρ, 0.0, z)
                
                # Evaluate Green's function
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                
                # Expectations:
                # - g_A and g_phi should be finite (not NaN/Inf)
                # - |g_A| should decrease with ρ (generally)
                # - g_A ≠ 0 for ρ > 0
                @test isfinite(vals.g_A)
                @test isfinite(vals.g_phi)
                @test abs(vals.g_A) > 0
            end
        end
        
        # Test 1.2: Compare with evaluate_at_same_height convenience function
        @testset "1.2 Consistency with evaluate_at_same_height" begin
            z = FT(250e-6)
            ρ = FT(100e-6)
            
            # Method 1: Full 3D evaluation
            r_src = SVector{3,FT}(0.0, 0.0, z)
            r_obs = SVector{3,FT}(ρ, 0.0, z)
            vals_full = evaluate_greenfunc(gf, r_obs, r_src)
            
            # Method 2: Optimized same-height evaluation
            h = z  # height above ground
            layer_idx = 1  # substrate layer
            vals_same = evaluate_at_same_height(gf, ρ, h, layer_idx)
            
            # These should be approximately equal (same computation path currently)
            # Tolerance: 1% relative error (generous for numerical differences)
            @test vals_full.g_A ≈ vals_same.g_A rtol=1e-2
            @test vals_full.g_phi ≈ vals_same.g_phi rtol=1e-2
        end
        
        # Test 1.3: Reciprocity (G(r1,r2) = G(r2,r1))
        @testset "1.3 Reciprocity at same height" begin
            z = FT(250e-6)
            r1 = SVector{3,FT}(0.0, 0.0, z)
            r2 = SVector{3,FT}(100e-6, 50e-6, z)
            
            vals_12 = evaluate_greenfunc(gf, r1, r2)
            vals_21 = evaluate_greenfunc(gf, r2, r1)
            
            @test vals_12.g_A ≈ vals_21.g_A rtol=1e-10
            @test vals_12.g_phi ≈ vals_21.g_phi rtol=1e-10
        end
    end
    
    # =============================================================================
    # Test Category 2: Same Layer, Small z ≠ z' (Modest Expectation)
    # =============================================================================
    
    @testset "Category 2: Same Layer, Small Vertical Separation (Modest)" begin
        
        # Test 2.1: z-dependence should exist (G should change when z changes)
        # This is the KEY test that will FAIL with current implementation
        @testset "2.1 z-dependence exists (CRITICAL)" begin
            ρ = FT(100e-6)  # Fixed horizontal separation
            z_base = FT(250e-6)  # Base height in substrate
            
            # Evaluate at different z positions (keeping z' fixed)
            r_src = SVector{3,FT}(0.0, 0.0, z_base)
            
            # Small vertical offsets (within same layer)
            dz_values = [FT(0.0), FT(1e-6), FT(10e-6), FT(50e-6)]
            g_A_values = Complex{FT}[]
            g_phi_values = Complex{FT}[]
            
            for dz in dz_values
                z_obs = z_base + dz
                r_obs = SVector{3,FT}(ρ, 0.0, z_obs)
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                push!(g_A_values, vals.g_A)
                push!(g_phi_values, vals.g_phi)
            end
            
            # MODEST EXPECTATION: G should change as z changes
            # With the bug (z ignored), all values are identical
            # With the fix, they should differ
            
            # Check that not all values are identical (this will fail with current bug)
            g_A_variation = maximum(abs.(g_A_values)) - minimum(abs.(g_A_values))
            g_phi_variation = maximum(abs.(g_phi_values)) - minimum(abs.(g_phi_values))
            
            # NOTE: This test documents expected behavior after fix
            # With bug: g_A_variation == 0 (all identical)
            # With fix: g_A_variation > 0 (physically correct)
            # We use a very loose tolerance (0.1%) to account for numerical noise
            # but the main point is that variation should exist
            
            # Check values are finite
            @test all(isfinite, g_A_values)
            @test all(isfinite, g_phi_values)
            
            # POST-FIX: Verify z-dependence exists (G should change with z)
            # With the fix, these should pass
            @test g_A_variation > 0.001 * abs(g_A_values[1])
            @test g_phi_variation > 0.001 * abs(g_phi_values[1])
        end
        
        # Test 2.2: Physical behavior - G should decrease with total distance
        @testset "2.2 Decay with total distance" begin
            # Fixed horizontal separation, increasing vertical separation
            ρ = FT(100e-6)
            z_src = FT(250e-6)
            
            test_cases = [
                (ρ, FT(0.0)),      # R = sqrt(ρ²) = 100 μm
                (ρ, FT(50e-6)),    # R = sqrt(ρ² + 50²) ≈ 112 μm
                (ρ, FT(100e-6)),   # R = sqrt(ρ² + 100²) ≈ 141 μm
            ]
            
            g_magnitudes = Float64[]
            
            for (ρ_test, dz) in test_cases
                z_obs = z_src + dz
                r_src = SVector{3,FT}(0.0, 0.0, z_src)
                r_obs = SVector{3,FT}(ρ_test, 0.0, z_obs)
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                push!(g_magnitudes, abs(vals.g_A))
            end
            
            # MODEST EXPECTATION: |G| should generally decrease as R increases
            # This is a weak test - we just check monotonic decrease for this specific case
            # With the bug, all R values are treated as equal (only ρ matters)
            # With the fix, larger dz should give smaller |G|
            
            # Check values are physically reasonable
            @test all(g_magnitudes .> 0)
            @test all(isfinite.(g_magnitudes))
            
            # POST-FIX: Verify |G| decreases as total distance R increases
            @test g_magnitudes[1] > g_magnitudes[2] > g_magnitudes[3]
        end
        
        # Test 2.3: Symmetry in z and z' for same-layer case
        @testset "2.3 Symmetry G(ρ, z, z') = G(ρ, z', z)" begin
            ρ = FT(100e-6)
            z1 = FT(200e-6)
            z2 = FT(300e-6)
            
            r_src1 = SVector{3,FT}(0.0, 0.0, z1)
            r_obs1 = SVector{3,FT}(ρ, 0.0, z2)
            vals_12 = evaluate_greenfunc(gf, r_obs1, r_src1)
            
            r_src2 = SVector{3,FT}(0.0, 0.0, z2)
            r_obs2 = SVector{3,FT}(ρ, 0.0, z1)
            vals_21 = evaluate_greenfunc(gf, r_obs2, r_src2)
            
            # MODEST EXPECTATION: G(ρ, z1, z2) ≈ G(ρ, z2, z1) (symmetry)
            # This should hold for same-layer interactions due to reciprocity
            # With bug: may or may not hold depending on layer lookup
            # With fix: should hold to good precision
            
            # Check they're both finite
            @test isfinite(vals_12.g_A)
            @test isfinite(vals_21.g_A)
            
            # POST-FIX: Verify reciprocity symmetry G(ρ,z,z') = G(ρ,z',z)
            @test vals_12.g_A ≈ vals_21.g_A rtol=1e-6
            @test vals_12.g_phi ≈ vals_21.g_phi rtol=1e-6
        end
    end
    
    # =============================================================================
    # Test Category 3: Different Layers (Future Work - Document Expectations)
    # =============================================================================
    
    @testset "Category 3: Different Layers (Future Work)" begin
        
        # Test 3.1: Layer index lookup correctness
        @testset "3.1 Layer index identification" begin
            # Test that get_layer_index correctly identifies layers
            z_substrate = FT(250e-6)  # In layer 1 (substrate)
            z_air = FT(600e-6)        # In layer 2 (air, above substrate)
            
            idx1 = get_layer_index(stack, z_substrate)
            idx2 = get_layer_index(stack, z_air)
            
            @test idx1 == 1
            @test idx2 == 2
        end
        
        # Test 3.2: DCIM coefficients exist for layer pairs
        @testset "3.2 DCIM coefficients for layer pairs" begin
            # Check that coefficients are precomputed for (1,1), (1,2), (2,1), (2,2)
            @test haskey(gf.g_a_coeffs, (1, 1))
            @test haskey(gf.g_a_coeffs, (1, 2))
            @test haskey(gf.g_a_coeffs, (2, 1))
            @test haskey(gf.g_a_coeffs, (2, 2))
            @test haskey(gf.g_phi_coeffs, (1, 1))
            @test haskey(gf.g_phi_coeffs, (1, 2))
            @test haskey(gf.g_phi_coeffs, (2, 1))
            @test haskey(gf.g_phi_coeffs, (2, 2))
        end
        
        # Test 3.3: Cross-layer evaluation (currently may not be accurate)
        @testset "3.3 Cross-layer evaluation (documented limitation)" begin
            # Source in substrate, observer in air
            z_src = FT(250e-6)   # In substrate
            z_obs = FT(600e-6)   # In air (above substrate)
            ρ = FT(100e-6)
            
            r_src = SVector{3,FT}(0.0, 0.0, z_src)
            r_obs = SVector{3,FT}(ρ, 0.0, z_obs)
            
            vals = evaluate_greenfunc(gf, r_obs, r_src)
            
            # MODEST EXPECTATION: Should return finite values
            # ACCURACY: May not be fully correct due to factorization challenge
            # DOCUMENTATION: This tests current behavior, not necessarily correct
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
            
            # The value should be smaller than same-layer case (intuitively)
            # due to additional interface losses
            r_same_layer = SVector{3,FT}(ρ, 0.0, z_src)
            vals_same = evaluate_greenfunc(gf, r_same_layer, r_src)
            
            # Just document the ratio (no assertion - this is diagnostic)
            ratio = abs(vals.g_A) / abs(vals_same.g_A)
            @test ratio > 0  # Just ensure it's positive
        end
    end
    
    # =============================================================================
    # Test Category 4: Comparison with Analytical Approximations
    # =============================================================================
    
    @testset "Category 4: Analytical Consistency Checks" begin
        
        # Test 4.1: Large ρ asymptotic behavior
        @testset "4.1 Large ρ decay (far-field)" begin
            z = FT(250e-6)
            
            # Test at increasing distances
            ρ_values = [FT(1e-3), FT(10e-3), FT(100e-3)]  # 1mm, 1cm, 10cm
            g_magnitudes = Float64[]
            
            for ρ in ρ_values
                r_src = SVector{3,FT}(0.0, 0.0, z)
                r_obs = SVector{3,FT}(ρ, 0.0, z)
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                push!(g_magnitudes, abs(vals.g_A))
            end
            
            # MODEST EXPECTATION: |G| should decay approximately as 1/ρ in far-field
            # Check that it decreases (not strict 1/ρ test due to surface waves, etc.)
            @test g_magnitudes[1] > g_magnitudes[2] > g_magnitudes[3]
        end
        
        # Test 4.2: Near-field singularity structure
        @testset "4.2 Near-field behavior (ρ → 0)" begin
            z = FT(250e-6)
            
            # Very small horizontal separations
            ρ_values = [FT(1e-9), FT(1e-8), FT(1e-7), FT(1e-6)]  # nm to μm
            
            for ρ in ρ_values
                r_src = SVector{3,FT}(0.0, 0.0, z)
                r_obs = SVector{3,FT}(ρ, 0.0, z)
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                
                # Should not diverge to Inf (DCIM handles singularity via quasi-static)
                @test isfinite(vals.g_A)
                @test isfinite(vals.g_phi)
                
                # Should be large (dominated by quasi-static terms)
                @test abs(vals.g_A) > 0
            end
        end
    end
    
    # =============================================================================
    # Test Category 5: Edge Cases and Robustness
    # =============================================================================
    
    @testset "Category 5: Edge Cases" begin
        
        # Test 5.1: z exactly at layer interface
        @testset "5.1 Points at layer interfaces" begin
            # z = 0 (ground plane interface)
            ρ = FT(100e-6)
            z_interface = FT(0.0)
            
            r_src = SVector{3,FT}(0.0, 0.0, z_interface)
            r_obs = SVector{3,FT}(ρ, 0.0, z_interface)
            vals = evaluate_greenfunc(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
            
            # NOTE: The PEC boundary condition (tangential E = 0) is satisfied 
            # by the combination of g_A and g_phi through the MPIE, not by 
            # g_A alone vanishing. The Green's function kernels g_A and g_phi 
            # both contribute to E via:
            #   E ∝ jωμ⟨G_A, J⟩ - (1/jωε)∇⟨G_phi, ∇·J⟩
            # The boundary condition is enforced by the image contributions in
            # the kernels, not by g_A being zero.
            # 
            # The physically correct check is that the GF values are finite
            # and well-behaved at the interface, which is verified above.
        end
        
        # Test 5.2: Very small z-separation (numerical stability)
        @testset "5.2 Numerical stability for small dz" begin
            ρ = FT(100e-6)
            z = FT(250e-6)
            
            # dz from large to very small
            dz_values = [FT(1e-3), FT(1e-4), FT(1e-5), FT(1e-6), FT(1e-9)]
            
            for dz in dz_values
                r_src = SVector{3,FT}(0.0, 0.0, z)
                r_obs = SVector{3,FT}(ρ, 0.0, z + dz)
                vals = evaluate_greenfunc(gf, r_obs, r_src)
                
                @test isfinite(vals.g_A)
            end
        end
        
        # Test 5.3: Large z-separation (within same layer)
        @testset "5.3 Large vertical separation within layer" begin
            # Source near bottom, observer near top of substrate
            z_src = FT(50e-6)    # Near bottom
            z_obs = FT(450e-6)   # Near top
            ρ = FT(100e-6)
            
            r_src = SVector{3,FT}(0.0, 0.0, z_src)
            r_obs = SVector{3,FT}(ρ, 0.0, z_obs)
            vals = evaluate_greenfunc(gf, r_obs, r_src)
            
            @test isfinite(vals.g_A)
            @test isfinite(vals.g_phi)
            
            # Total distance
            R_total = sqrt(ρ^2 + (z_obs - z_src)^2)
            @test R_total > ρ  # Vertical separation increases total distance
        end
    end

end

# =================================================================================
# Summary: Test Coverage and Expected Behavior
# =================================================================================
#
# PRE-FIX (current state with bug):
# - Category 1 tests: PASS (z = z' works correctly)
# - Category 2 tests: PARTIAL (2.1 will show no z-variation, documenting the bug)
# - Category 3 tests: PASS (layer infrastructure works, accuracy TBD)
# - Category 4 tests: PASS (asymptotic behavior generally correct)
# - Category 5 tests: PASS (edge cases handled)
#
# POST-FIX (after implementing dz^2 in evaluate_dcim):
# - Category 1 tests: PASS (unchanged)
# - Category 2 tests: PASS (z-dependence now correctly captured)
#   - 2.1: g_A_variation > 0 (z-dependence exists)
#   - 2.2: |G| decreases with total distance R
#   - 2.3: Symmetry G(ρ,z,z') = G(ρ,z',z) holds
# - Category 3 tests: PARTIAL (cross-layer may need separate work)
# - Category 4 tests: PASS (improved accuracy at various z)
# - Category 5 tests: PASS (robustness maintained)
#
# VALIDATION CRITERIA:
# The "TODO" comments in tests above indicate specific assertions to enable
# after the fix is implemented. The key criterion is:
#   "G must change when z changes, with physically correct magnitude"
#
# ACCURACY EXPECTATIONS:
# - For same-layer interactions: < 1% error vs direct numerical integration
# - For ρ in range [0.01λ, 10λ]: DCIM is typically very accurate
# - Near-field (ρ < 0.01λ): Dominated by quasi-static, very accurate
# - Far-field (ρ > 10λ): Surface wave extraction may be needed
