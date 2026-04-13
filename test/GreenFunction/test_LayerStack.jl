# test_LayerStack.jl
# Unit tests for LayerStack data structures

@testset "LayerStack Data Structures" begin

    @testset "LayerInfo Construction" begin
        # Basic layer
        layer = LayerInfo("substrate", 11.7, 1.0, 500e-6)
        @test layer.name == "substrate"
        @test layer.eps_r ≈ 11.7 atol=1e-14
        @test layer.mu_r ≈ 1.0 atol=1e-14
        @test layer.thickness ≈ 500e-6 atol=1e-14
        
        # Half-space layer
        layer_air = LayerInfo("air", 1.0, 1.0, Inf)
        @test is_halfspace(layer_air) == true
        @test is_halfspace(layer) == false
        
        # Complex permittivity
        layer_lossy = LayerInfo("lossy", 4.3 - 0.1im, 1.0, 1e-3)
        @test real(layer_lossy.eps_r) ≈ 4.3 atol=1e-14
        @test imag(layer_lossy.eps_r) ≈ -0.1 atol=1e-14
    end

    @testset "LayerStack Construction" begin
        layers = [
            LayerInfo("substrate", 11.7, 1.0, 500e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0)
        
        @test stack.n_layers == 2
        @test stack.reference_z ≈ 0.0 atol=1e-14
        
        # Interface positions
        # Layer 1: 0.0 to 500e-6
        # Layer 2: 500e-6 to Inf
        @test stack.interfaces[1] ≈ 0.0 atol=1e-14
        @test stack.interfaces[2] ≈ 500e-6 atol=1e-14
        @test isinf(stack.interfaces[3])
    end

    @testset "Layer Index Lookup" begin
        layers = [
            LayerInfo("substrate", 11.7, 1.0, 500e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0)
        
        # In substrate (layer 1)
        @test get_layer_index(stack, 0.0) == 1
        @test get_layer_index(stack, 100e-6) == 1
        @test get_layer_index(stack, 499e-6) == 1
        
        # In air (layer 2)
        @test get_layer_index(stack, 500e-6) == 2
        @test get_layer_index(stack, 600e-6) == 2
        @test get_layer_index(stack, 1.0) == 2
    end

    @testset "Same Layer Detection" begin
        layers = [
            LayerInfo("substrate", 11.7, 1.0, 500e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0)
        
        # Same layer
        @test are_in_same_layer(stack, 100e-6, 200e-6) == true
        @test are_in_same_layer(stack, 600e-6, 700e-6) == true
        
        # Different layers
        @test are_in_same_layer(stack, 100e-6, 600e-6) == false
        @test are_in_same_layer(stack, 400e-6, 600e-6) == false
    end

    @testset "Layer Access" begin
        layers = [
            LayerInfo("substrate", 11.7, 1.0, 500e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0)
        
        # Get layer by index
        @test get_layer(stack, 100e-6).name == "substrate"
        @test get_layer(stack, 600e-6).name == "air"
        
        # Get layer by z-coordinate
        layer1 = get_layer(stack, 200e-6)
        @test layer1.name == "substrate"
        @test layer1.eps_r ≈ 11.7 atol=1e-14
    end

    @testset "Layer Thickness Queries" begin
        layers = [
            LayerInfo("substrate", 11.7, 1.0, 500e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0)
        
        # In substrate at z = 200e-6
        # Below: 200e-6 from reference
        # Above: 500e-6 - 200e-6 = 300e-6 to interface
        z = 200e-6
        @test layer_thickness_below(stack, z) ≈ 200e-6 atol=1e-14
        @test layer_thickness_above(stack, z) ≈ 300e-6 atol=1e-14
    end

    @testset "Multi-Layer Stack" begin
        # More complex stack - note: no zero-thickness layers allowed
        layers = [
            LayerInfo("oxide1", 3.9, 1.0, 0.5e-6),
            LayerInfo("metal", 1.0, 1.0, 0.35e-6),
            LayerInfo("oxide2", 3.9, 1.0, 0.85e-6),
            LayerInfo("air", 1.0, 1.0, Inf)
        ]
        stack = LayerStack(layers; reference_z=0.0)
        
        @test stack.n_layers == 4
        
        # Check interfaces add up correctly
        @test stack.interfaces[1] ≈ 0.0 atol=1e-14
        @test stack.interfaces[2] ≈ 0.5e-6 atol=1e-14
        @test stack.interfaces[3] ≈ 0.85e-6 atol=1e-14
        @test stack.interfaces[4] ≈ 1.7e-6 atol=1e-14
        @test isinf(stack.interfaces[5])
    end

    @testset "Effective Permittivity" begin
        layer_lossless = LayerInfo("lossless", 11.7, 1.0, 500e-6, 0.0)
        
        # No conductivity
        eps_eff = effective_permittivity(layer_lossless, 1e9)
        @test eps_eff ≈ 11.7 atol=1e-14
        
        # With conductivity
        layer_lossy = LayerInfo("lossy", 11.7, 1.0, 500e-6, 10.0)  # sigma = 10 S/m
        eps_eff_lossy = effective_permittivity(layer_lossy, 1e9)
        @test real(eps_eff_lossy) ≈ 11.7 atol=1e-14
        @test imag(eps_eff_lossy) < 0  # Loss adds negative imaginary part
    end

end
