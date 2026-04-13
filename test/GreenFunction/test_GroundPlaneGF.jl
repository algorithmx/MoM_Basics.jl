# test_GroundPlaneGF.jl
# Unit tests for GroundPlaneGF - Image Theory Mathematical Correctness
#
# Tests verify the equations from mom_ground_plane_dielectric_approaches.md Section 1.2:
#   G_A(r, r')   = G_free(r, r') - G_free(r, r'_image)
#   G_phi(r, r') = G_free(r, r') + G_free(r, r'_image)
# where r'_image = (x', y', 2*z_gnd - z')

@testset "GroundPlaneGF Mathematical Correctness" begin

    @testset "Image Point Geometry: r_img = (x, y, 2*z_gnd - z)" begin
        # Ground at z = 0
        z_gnd = 0.0
        r_src = [1.0, 2.0, 3.0]
        r_img_expected = [1.0, 2.0, -3.0]
        
        r_img = mirror_point_across_ground(r_src, z_gnd)
        @test r_img ≈ r_img_expected atol=1e-14
        
        # Ground at z = 1.0
        z_gnd = 1.0
        r_src = [1.0, 2.0, 3.0]
        r_img_expected = [1.0, 2.0, -1.0]  # 2*1.0 - 3.0 = -1.0
        
        r_img = mirror_point_across_ground(r_src, z_gnd)
        @test r_img ≈ r_img_expected atol=1e-14
        
        # Ground at z = -2.0
        z_gnd = -2.0
        r_src = [0.0, 0.0, 0.0]
        r_img_expected = [0.0, 0.0, -4.0]  # 2*(-2.0) - 0.0 = -4.0
        
        r_img = mirror_point_across_ground(r_src, z_gnd)
        @test r_img ≈ r_img_expected atol=1e-14
    end

    @testset "CRITICAL: Sign Verification - Both A and phi use (-)" begin
        # For a horizontal source, image current is reversed so A has (-)
        # Consequently, the image charge has opposite sign, so phi ALSO has (-)
        
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        # Source at height h, observer at height 2h
        h = 1.0
        r_src = [0.0, 0.0, h]
        r_obs = [0.0, 0.0, 2*h]
        
        # Direct distance = h (vertical separation)
        R_direct = h
        
        # Image is at (0, 0, -h), so image distance = 3h
        r_src_img = mirror_point_across_ground(r_src, z_gnd)
        @test r_src_img[3] ≈ -h atol=1e-14
        R_image = norm(r_obs - r_src_img)  # Should be 3h
        @test R_image ≈ 3*h atol=1e-14
        
        # Expected Green's functions
        G_direct = exp(-im*k*R_direct) / R_direct
        G_image = exp(-im*k*R_image) / R_image
        
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        
        # CRITICAL VERIFICATION: Both subtract image
        @test vals.g_A ≈ G_direct - G_image atol=1e-14
        @test vals.g_phi ≈ G_direct - G_image atol=1e-14
        
        # Verify they are the same
        @test vals.g_A ≈ vals.g_phi atol=1e-14
    end

    @testset "Sign Verification at Multiple Distances" begin
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        test_cases = [
            (1.0, 2.0),   # h=1, obs at 2
            (0.5, 1.0),   # h=0.5, obs at 1.0
            (2.0, 4.0),   # h=2, obs at 4
            (1.0, 1.5),   # h=1, obs at 1.5 (closer)
        ]
        
        for (h, z_obs) in test_cases
            r_src = [0.0, 0.0, h]
            r_obs = [0.0, 0.0, z_obs]
            
            r_src_img = mirror_point_across_ground(r_src, z_gnd)
            
            R_direct = norm(r_obs - r_src)
            R_image = norm(r_obs - r_src_img)
            
            G_direct = exp(-im*k*R_direct) / R_direct
            G_image = exp(-im*k*R_image) / R_image
            
            vals = evaluate_greenfunc(gf, r_obs, r_src)
            
            @test vals.g_A ≈ G_direct - G_image atol=1e-12
            @test vals.g_phi ≈ G_direct - G_image atol=1e-12
        end
    end

    @testset "PEC Boundary Condition: Tangential A vanishes at z = z_gnd" begin
        # At ground plane surface, tangential E should vanish
        # This requires g_A = 0 when observer is on ground plane
        
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        # Observer on ground plane
        r_obs = [1.0, 1.0, z_gnd]
        # Source above ground
        r_src = [0.0, 0.0, 1.0]
        
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        
        # At ground plane: |r_obs - r_src| = |r_obs - r_src_image|
        r_src_img = mirror_point_across_ground(r_src, z_gnd)
        R_direct = norm(r_obs - r_src)
        R_image = norm(r_obs - r_src_img)
        
        @test R_direct ≈ R_image atol=1e-14
        
        # Therefore: g_A = G - G = 0 at ground plane
        @test abs(vals.g_A) < 1e-14
        
        # And g_phi = G - G = 0 at ground plane (to ensure E_tangential = 0)
        G = exp(-im*k*R_direct) / R_direct
        @test abs(vals.g_phi) < 1e-14
    end

    @testset "Same-Height Evaluation" begin
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        h = 1.0  # height above ground
        ρ = 2.0  # horizontal separation
        
        r_src = [0.0, 0.0, h]
        r_obs = [ρ, 0.0, h]
        
        # General evaluation
        vals_general = evaluate_greenfunc(gf, r_obs, r_src)
        
        # Same-height optimized evaluation
        vals_same = evaluate_at_same_height(gf, ρ, h)
        
        @test vals_same.g_A ≈ vals_general.g_A atol=1e-14
        @test vals_same.g_phi ≈ vals_general.g_phi atol=1e-14
        
        # Verify the distances used
        R_direct = ρ  # same height
        R_image = sqrt(ρ^2 + (2*h)^2)  # via image
        
        G_direct = exp(-im*k*R_direct) / R_direct
        G_image = exp(-im*k*R_image) / R_image
        
        @test vals_same.g_A ≈ G_direct - G_image atol=1e-14
        @test vals_same.g_phi ≈ G_direct - G_image atol=1e-14
    end

    @testset "Reciprocity: G(r1, r2) = G(r2, r1) with ground plane" begin
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        # Points above ground
        r1 = [1.0, 0.0, 2.0]
        r2 = [-0.5, 1.0, 1.5]
        
        vals_12 = evaluate_greenfunc(gf, r1, r2)
        vals_21 = evaluate_greenfunc(gf, r2, r1)
        
        # Both A and phi should be reciprocal
        @test vals_12.g_A ≈ vals_21.g_A atol=1e-14
        @test vals_12.g_phi ≈ vals_21.g_phi atol=1e-14
    end

    @testset "Height Above Ground Functions" begin
        z_gnd = 1.0
        k = Complex(2π)
        gf = GroundPlaneGF(k, z_gnd)
        
        # Point at z = 3, ground at z = 1
        r = [0.0, 0.0, 3.0]
        @test height_above_ground(gf, r) ≈ 2.0 atol=1e-14
        @test is_above_ground(gf, r) == true
        
        # Point exactly on ground
        r = [0.0, 0.0, 1.0]
        @test height_above_ground(gf, r) ≈ 0.0 atol=1e-14
        @test is_above_ground(gf, r) == true
    end

    @testset "Direct and Image Component Access" begin
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        r_src = SVector{3, Float64}(0.0, 0.0, 1.0)
        r_obs = SVector{3, Float64}(1.0, 0.0, 2.0)
        
        # Get individual components
        G_direct = evaluate_greenfunc_direct(gf, r_obs, r_src)
        G_image = evaluate_greenfunc_image(gf, r_obs, r_src)
        
        # Verify they sum correctly
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        
        @test vals.g_A ≈ G_direct - G_image atol=1e-14
        @test vals.g_phi ≈ G_direct - G_image atol=1e-14
    end

    @testset "Display and Summary" begin
        k = Complex(2π)
        z_gnd = 0.0
        gf = GroundPlaneGF(k, z_gnd)
        
        # Test that display functions work without error
        @test sprint(show, gf) isa String
        @test sprint(summary, gf) isa String
        
        # Should contain ground plane info
        str = sprint(show, gf)
        @test occursin("GroundPlaneGF", str)
    end

end
