# =============================================================================
# Port Masking Tests (Logical - Non-Destructive)
# =============================================================================
#
# Tests for logical port masking that identifies port regions without
# modifying the mesh structure.
#
# =============================================================================

@testset "PortMasking" begin
    
    # Setup: Create a simple planar mesh with finer resolution
    function _build_test_mesh(nx::Int=9, ny::Int=9)
        xs = range(0.0, 1.0, length=nx)
        ys = range(0.0, 1.0, length=ny)
        
        nodes = zeros(Float64, 3, nx * ny)
        for j in 1:ny
            for i in 1:nx
                idx = (j - 1) * nx + i
                nodes[:, idx] = [Float64(xs[i]), Float64(ys[j]), 0.0]
            end
        end
        
        triangles = Matrix{Int}(undef, 3, 2 * (nx - 1) * (ny - 1))
        tri_idx = 1
        for j in 1:(ny - 1)
            for i in 1:(nx - 1)
                n1 = (j - 1) * nx + i
                n2 = n1 + 1
                n3 = j * nx + i
                n4 = n3 + 1
                
                triangles[:, tri_idx] = [n1, n2, n4]
                tri_idx += 1
                triangles[:, tri_idx] = [n1, n4, n3]
                tri_idx += 1
            end
        end
        
        mesh = TriangleMesh(size(triangles, 2), nodes, triangles)
        return MoM_Basics.getTriangleInfo(mesh; keep_half_rwg=true)
    end
    
    @testset "PortMask Creation" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        # Define a rectangular port region (center of the mesh, larger to capture triangles)
        center = MVec3D{Float64}(0.5, 0.5, 0.0)
        normal = MVec3D{Float64}(0.0, 0.0, 1.0)
        width, height = 0.5, 0.5  # Larger region to ensure triangles are captured
        
        # Create predicate for rectangular region
        predicate = p -> abs(p[1] - 0.5) <= width/2 && abs(p[2] - 0.5) <= height/2
        
        # Create port mask
        mask = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo)
        
        # Verify mask structure
        @test typeof(mask) <: PortMask
        @test !isempty(mask.vertexIDs)
        @test !isempty(mask.triangleIDs)
        @test !isempty(mask.boundaryEdgeIDs)
        @test mask.maskType == :aperture
        
        # Verify all IDs are valid
        num_vertices = 9 * 9  # 9x9 grid
        @test all(vid -> 1 <= vid <= num_vertices, mask.vertexIDs)
        @test all(tid -> 1 <= tid <= length(trianglesInfo), mask.triangleIDs)
        @test all(eid -> 1 <= eid <= length(rwgsInfo), mask.boundaryEdgeIDs)
    end
    
    @testset "PortMask from DeltaGapArrayPort" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        # Create and bind a DeltaGapArrayPort
        port = DeltaGapArrayPort{Float64, Int}(
            id = 1,
            center = MVec3D{Float64}(0.5, 0.5, 0.0),
            normal = MVec3D{Float64}(0.0, 0.0, 1.0),
            excitationDistribution = UniformDistribution{Float64}()
        )
        
        predicate = p -> abs(p[1] - 0.5) <= 0.2 && abs(p[2] - 0.5) <= 0.2
        bind_to_mesh!(port, predicate, trianglesInfo, rwgsInfo)
        
        @test port.isBound == true
        
        # Create mask from bound port
        mask = create_port_mask(port, trianglesInfo, rwgsInfo)
        
        # Verify mask matches port data
        @test sort(mask.vertexIDs) == sort(port.vertexIDs)
        @test sort(mask.triangleIDs) == sort(port.triangleIDs)
        @test sort(mask.boundaryEdgeIDs) == sort(port.rwgIDs)
    end
    
    @testset "PortMask Validation" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        center = MVec3D{Float64}(0.5, 0.5, 0.0)
        normal = MVec3D{Float64}(0.0, 0.0, 1.0)
        predicate = p -> abs(p[1] - 0.5) <= 0.2 && abs(p[2] - 0.5) <= 0.2
        
        mask = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo)
        
        # Test validation
        is_valid, issues = validate_port_mask(mask, trianglesInfo, rwgsInfo)
        @test is_valid == true
        @test isempty(issues)
        
        # Test with verbose output (should not error)
        is_valid2, _ = validate_port_mask(mask, trianglesInfo, rwgsInfo; verbose=true)
        @test is_valid2 == true
    end
    
    @testset "PortMask Statistics" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        center = MVec3D{Float64}(0.5, 0.5, 0.0)
        normal = MVec3D{Float64}(0.0, 0.0, 1.0)
        predicate = p -> abs(p[1] - 0.5) <= 0.2 && abs(p[2] - 0.5) <= 0.2
        
        mask = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo)
        stats = get_mask_statistics(mask, trianglesInfo)
        
        @test stats.num_vertices > 0
        @test stats.num_triangles > 0
        @test stats.num_boundary_edges > 0
        @test stats.mask_type == :aperture
    end
    
    @testset "PortMask Geometry Extraction" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        center = MVec3D{Float64}(0.5, 0.5, 0.0)
        normal = MVec3D{Float64}(0.0, 0.0, 1.0)
        predicate = p -> abs(p[1] - 0.5) <= 0.2 && abs(p[2] - 0.5) <= 0.2
        
        mask = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo)
        geom = get_mask_geometry(mask, trianglesInfo, rwgsInfo)
        
        @test !isempty(geom.aperture_vertices)
        @test !isempty(geom.aperture_tri_centers)
        @test !isempty(geom.boundary_edge_centers)
        @test length(geom.boundary_edge_centers) == length(geom.boundary_edge_orients)
    end
    
    @testset "PortMaskCollection" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        # Create two non-overlapping ports
        port1 = DeltaGapArrayPort{Float64, Int}(
            id = 1,
            center = MVec3D{Float64}(0.25, 0.5, 0.0),
            normal = MVec3D{Float64}(0.0, 0.0, 1.0),
            excitationDistribution = UniformDistribution{Float64}()
        )
        
        port2 = DeltaGapArrayPort{Float64, Int}(
            id = 2,
            center = MVec3D{Float64}(0.75, 0.5, 0.0),
            normal = MVec3D{Float64}(0.0, 0.0, 1.0),
            excitationDistribution = UniformDistribution{Float64}()
        )
        
        # Bind ports
        predicate1 = p -> abs(p[1] - 0.25) <= 0.15 && abs(p[2] - 0.5) <= 0.15
        predicate2 = p -> abs(p[1] - 0.75) <= 0.15 && abs(p[2] - 0.5) <= 0.15
        
        bind_to_mesh!(port1, predicate1, trianglesInfo, rwgsInfo)
        bind_to_mesh!(port2, predicate2, trianglesInfo, rwgsInfo)
        
        # Create collection
        ports = [port1, port2]
        collection = create_port_mask_collection(ports, trianglesInfo, rwgsInfo)
        
        @test typeof(collection) <: PortMaskCollection
        @test length(collection.masks) == 2
        @test haskey(collection.masks, 1)
        @test haskey(collection.masks, 2)
        
        # Test overlap checking (should be no overlap)
        has_overlap, overlaps = check_mask_overlap(collection)
        @test has_overlap == false
        @test isempty(overlaps)
    end
    
    @testset "PortMask Error Handling" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        # Test with empty predicate (no vertices found)
        empty_predicate = p -> false
        center = MVec3D{Float64}(0.5, 0.5, 0.0)
        normal = MVec3D{Float64}(0.0, 0.0, 1.0)
        
        @test_throws ErrorException create_port_mask(
            empty_predicate, center, normal, trianglesInfo, rwgsInfo
        )
        
        # Test unbound port
        unbound_port = DeltaGapArrayPort{Float64, Int}(
            id = 1,
            center = MVec3D{Float64}(0.5, 0.5, 0.0),
            normal = MVec3D{Float64}(0.0, 0.0, 1.0),
            excitationDistribution = UniformDistribution{Float64}()
        )
        @test_throws ErrorException create_port_mask(unbound_port, trianglesInfo, rwgsInfo)
    end
    
    @testset "Different Mask Types" begin
        trianglesInfo, rwgsInfo = _build_test_mesh(9, 9)
        
        center = MVec3D{Float64}(0.5, 0.5, 0.0)
        normal = MVec3D{Float64}(0.0, 0.0, 1.0)
        predicate = p -> abs(p[1] - 0.5) <= 0.2 && abs(p[2] - 0.5) <= 0.2
        
        # Test different mask types
        mask_aperture = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo; maskType=:aperture)
        mask_boundary = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo; maskType=:boundary)
        mask_full = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo; maskType=:full)
        
        @test mask_aperture.maskType == :aperture
        @test mask_boundary.maskType == :boundary
        @test mask_full.maskType == :full
        
        # Invalid mask type should throw
        @test_throws AssertionError create_port_mask(
            predicate, center, normal, trianglesInfo, rwgsInfo; maskType=:invalid
        )
    end
    
end

