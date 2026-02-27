proj_path = joinpath(@__DIR__, "..")

@testset "triangleConnectivity basic invariants" begin
    filename = joinpath(proj_path, "meshfiles/Tri.nas")
    meshData, _ = getMeshData(filename; meshUnit = :mm)
    _, _, geosInfo, _ = getBFsFromMeshData(meshData)

    @test geosInfo isa AbstractVector{<:TriangleInfo}

    conn = triangleConnectivity(geosInfo)

    n = length(geosInfo)
    @test size(conn.tri_vertices, 1) == n
    @test size(conn.tri_vertices, 2) == 3

    @test size(conn.node_coords, 2) == 3
    @test size(conn.node_coords, 1) >= maximum(conn.tri_vertices)

    @test length(conn.edge_neighbors_indptr) == n + 1
    @test conn.edge_neighbors_indptr[1] == 1
    @test conn.edge_neighbors_indptr[end] == length(conn.edge_neighbors_indices) + 1

    @test length(conn.vertex_neighbors_indptr) == n + 1
    @test conn.vertex_neighbors_indptr[1] == 1
    @test conn.vertex_neighbors_indptr[end] == length(conn.vertex_neighbors_indices) + 1

    # Symmetry sanity: if j is in i's edge-neighbors, i should be in j's edge-neighbors.
    for i in 1:n
        a = conn.edge_neighbors_indptr[i]
        b = conn.edge_neighbors_indptr[i + 1] - 1
        for j in conn.edge_neighbors_indices[a:b]
            a2 = conn.edge_neighbors_indptr[j]
            b2 = conn.edge_neighbors_indptr[j + 1] - 1
            @test any(==(i), conn.edge_neighbors_indices[a2:b2])
        end
    end

    @test conn.tri_index_base == 1
    @test conn.node_index_base == 1
end
