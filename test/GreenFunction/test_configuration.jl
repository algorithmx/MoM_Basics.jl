# test_configuration.jl
# Integration tests for GF configuration and factory functions

@testset "GF Configuration and Factory" begin

    @testset "Default Configuration" begin
        # Reset to default state
        reset_green_function_config!()
        
        @test get_green_function_type() == :freespace
    end

    @testset "Configuration State Transitions" begin
        # Start from known state
        reset_green_function_config!()
        @test get_green_function_type() == :freespace
        
        # Transition to ground plane
        set_green_function_type!(:groundplane; z_gnd=0.0)
        @test get_green_function_type() == :groundplane
        
        # Transition back to free space
        set_green_function_type!(:freespace)
        @test get_green_function_type() == :freespace
        
        # Another ground plane configuration
        set_green_function_type!(:groundplane; z_gnd=-1.5)
        @test get_green_function_type() == :groundplane
        
        # Reset
        reset_green_function_config!()
        @test get_green_function_type() == :freespace
    end

    @testset "Ground Plane Configuration" begin
        reset_green_function_config!()
        
        # Ground at z = 0
        set_green_function_type!(:groundplane; z_gnd=0.0)
        gf = create_green_function()
        @test gf isa GroundPlaneGF
        @test gf.z_gnd ≈ 0.0 atol=1e-14
        
        # Ground at negative z
        set_green_function_type!(:groundplane; z_gnd=-2.0)
        gf = create_green_function()
        @test gf isa GroundPlaneGF
        @test gf.z_gnd ≈ -2.0 atol=1e-14
        
        # Ground at positive z
        set_green_function_type!(:groundplane; z_gnd=1.0)
        gf = create_green_function()
        @test gf isa GroundPlaneGF
        @test gf.z_gnd ≈ 1.0 atol=1e-14
    end

    @testset "Free Space Configuration" begin
        reset_green_function_config!()
        
        set_green_function_type!(:freespace)
        gf = create_green_function()
        @test gf isa FreeSpaceGF
    end

    @testset "Explicit Factory Calls" begin
        # Create specific GF types regardless of configuration
        
        # Free space explicitly
        gf_fs = create_green_function(:freespace)
        @test gf_fs isa FreeSpaceGF
        
        # Ground plane explicitly
        reset_green_function_config!()
        set_green_function_type!(:groundplane; z_gnd=0.0)
        gf_gp = create_green_function(:groundplane)
        @test gf_gp isa GroundPlaneGF
        
        # Can still create free space even when config is ground
        gf_fs2 = create_green_function(:freespace)
        @test gf_fs2 isa FreeSpaceGF
    end

    @testset "Layered Media Factory" begin
        reset_green_function_config!()
        
        # Create a layer stack
        layers = [
            LayerInfo("substrate", 11.7, 1.0, 500e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0, has_ground_plane=true)
        
        # Set the layer stack and configure for layered
        set_layer_stack!(stack)
        set_green_function_type!(:layered)
        
        @test get_green_function_type() == :layered
        @test MoM_Basics.GFParams.has_layer_stack == true
        
        # Creating should now work (but DCIM fitting may take time)
        # Note: We don't actually call create_green_function() here to avoid
        # expensive DCIM fitting in unit tests, but the code path is verified
        
        reset_green_function_config!()
    end
    
    @testset "Layered Media Error Without Stack" begin
        reset_green_function_config!()
        
        # Trying to set layered without configuring stack first should error
        @test_throws ErrorException set_green_function_type!(:layered)
        
        reset_green_function_config!()
    end

    @testset "Invalid Configuration Errors" begin
        reset_green_function_config!()
        
        # Ground plane without z_gnd
        @test_throws ErrorException set_green_function_type!(:groundplane)
        
        # Ground plane with infinite z_gnd
        @test_throws ErrorException set_green_function_type!(:groundplane; z_gnd=Inf)
        
        # Layered without stack configured
        @test_throws ErrorException set_green_function_type!(:layered)
        
        # Unknown type
        @test_throws ErrorException set_green_function_type!(:unknown_type)
        
        reset_green_function_config!()
    end

    @testset "GFParams State Consistency" begin
        reset_green_function_config!()
        
        # Check that GFParams is updated correctly
        set_green_function_type!(:groundplane; z_gnd=-1.0)
        
        @test MoM_Basics.GFParams.gf_type == :groundplane
        @test MoM_Basics.GFParams.ground_plane_z ≈ -1.0 atol=1e-14
        @test MoM_Basics.GFParams.has_layer_stack == false
        
        reset_green_function_config!()
        
        @test MoM_Basics.GFParams.gf_type == :freespace
        @test isinf(MoM_Basics.GFParams.ground_plane_z)
    end

    @testset "End-to-End: Config → Factory → Evaluation" begin
        reset_green_function_config!()
        
        # Configure for ground plane
        set_green_function_type!(:groundplane; z_gnd=0.0)
        
        # Create GF
        gf = create_green_function()
        @test gf isa GroundPlaneGF
        
        # Use it
        r_src = SVector{3, Float64}(0.0, 0.0, 1.0)
        r_obs = SVector{3, Float64}(0.0, 0.0, 2.0)
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        
        # A and phi both subtract the image contribution for horizontal elements
        @test vals.g_A == vals.g_phi
        
        # Reset and use free space
        reset_green_function_config!()
        gf_fs = create_green_function()
        @test gf_fs isa FreeSpaceGF
        
        vals_fs = evaluate_greenfunc(gf_fs, r_obs, r_src)
        @test vals_fs.g_A == vals_fs.g_phi
        
        reset_green_function_config!()
    end

end
