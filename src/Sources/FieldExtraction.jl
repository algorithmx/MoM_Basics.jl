"""
    FieldData{FT, CT}

Unified structure to store field data (incident fields, currents, etc.) at specific points.
"""
struct FieldData{FT<:AbstractFloat, CT<:Complex{FT}}
    npoints     ::Int
    positions   ::Vector{SVec3D{FT}} # centroid positions
    fields      ::Dict{Symbol, Vector{SVec3D{CT}}}
end

toFieldData(nt::NamedTuple) = FieldData{eltype(nt.positions[1]), eltype(nt.fields[:J][1])}(nt.npoints, nt.positions, nt.fields)


"""
    triangleConnectivity(tris::AbstractVector{<:TriangleInfo})

Build triangle connectivity using the current in-memory triangle ordering.

Returns a NamedTuple with:
- `tri_vertices` :: Matrix{Int64} (N×3) vertex IDs per triangle
- `node_coords` :: Matrix{Float64} (Nnodes×3) node coordinate table, indexed by node ID (1-based)
- `edge_neighbors_indptr`, `edge_neighbors_indices` :: CSR adjacency for edge-sharing
- `vertex_neighbors_indptr`, `vertex_neighbors_indices` :: CSR adjacency for vertex-sharing
- `tri_index_base` :: Int64 (always 1)
- `node_index_base` :: Int64 (always 1)

Neighbor indices are 1-based triangle indices in `tris`.
"""
function triangleConnectivity(tris::AbstractVector{<:TriangleInfo})
    n = length(tris)

    tri_vertices = Matrix{Int64}(undef, n, 3)
    # Build a dense node coordinate table indexed by node ID (1-based local IDs).
    max_vid = Int64(0)
    @inbounds for i in 1:n
        vids = tris[i].verticesID
        v1 = Int64(vids[1]); v2 = Int64(vids[2]); v3 = Int64(vids[3])
        max_vid = max(max_vid, v1, v2, v3)
    end
    node_coords = Matrix{Float64}(undef, max_vid, 3)
    filled = falses(max_vid)

    edge_to_tris = Dict{Tuple{Int64, Int64}, Vector{Int64}}()
    vert_to_tris = Dict{Int64, Vector{Int64}}()

    @inbounds for i in 1:n
        vids = tris[i].verticesID
        v1 = Int64(vids[1]); v2 = Int64(vids[2]); v3 = Int64(vids[3])
        tri_vertices[i, 1] = v1
        tri_vertices[i, 2] = v2
        tri_vertices[i, 3] = v3

        # Populate node coordinates from triangle vertex coordinates.
        # Each TriangleInfo stores vertices[:, local] in the same order as verticesID[local].
        @views begin
            if !filled[v1]
                node_coords[v1, 1] = Float64(tris[i].vertices[1, 1])
                node_coords[v1, 2] = Float64(tris[i].vertices[2, 1])
                node_coords[v1, 3] = Float64(tris[i].vertices[3, 1])
                filled[v1] = true
            end
            if !filled[v2]
                node_coords[v2, 1] = Float64(tris[i].vertices[1, 2])
                node_coords[v2, 2] = Float64(tris[i].vertices[2, 2])
                node_coords[v2, 3] = Float64(tris[i].vertices[3, 2])
                filled[v2] = true
            end
            if !filled[v3]
                node_coords[v3, 1] = Float64(tris[i].vertices[1, 3])
                node_coords[v3, 2] = Float64(tris[i].vertices[2, 3])
                node_coords[v3, 3] = Float64(tris[i].vertices[3, 3])
                filled[v3] = true
            end
        end

        for v in (v1, v2, v3)
            push!(get!(vert_to_tris, v, Int64[]), Int64(i))
        end

        for (a, b) in ((v1, v2), (v2, v3), (v3, v1))
            e = a < b ? (a, b) : (b, a)
            push!(get!(edge_to_tris, e, Int64[]), Int64(i))
        end
    end

    edge_neighbors_indptr = Vector{Int64}(undef, n + 1)
    edge_neighbors_indptr[1] = 1
    edge_neighbors_indices = Int64[]

    vertex_neighbors_indptr = Vector{Int64}(undef, n + 1)
    vertex_neighbors_indptr[1] = 1
    vertex_neighbors_indices = Int64[]

    @inbounds for i in 1:n
        v1 = tri_vertices[i, 1]
        v2 = tri_vertices[i, 2]
        v3 = tri_vertices[i, 3]

        # Edge-sharing neighbors
        neigh_edge = Int64[]
        seen_edge = Set{Int64}()
        for (a, b) in ((v1, v2), (v2, v3), (v3, v1))
            e = a < b ? (a, b) : (b, a)
            for t in edge_to_tris[e]
                t == i && continue
                if !(t in seen_edge)
                    push!(neigh_edge, t)
                    push!(seen_edge, t)
                end
            end
        end
        sort!(neigh_edge)
        append!(edge_neighbors_indices, neigh_edge)
        edge_neighbors_indptr[i + 1] = length(edge_neighbors_indices) + 1

        # Vertex-sharing neighbors
        neigh_vert = Int64[]
        seen_vert = Set{Int64}()
        for v in (v1, v2, v3)
            for t in vert_to_tris[v]
                t == i && continue
                if !(t in seen_vert)
                    push!(neigh_vert, t)
                    push!(seen_vert, t)
                end
            end
        end
        sort!(neigh_vert)
        append!(vertex_neighbors_indices, neigh_vert)
        vertex_neighbors_indptr[i + 1] = length(vertex_neighbors_indices) + 1
    end

    return (
        tri_vertices = tri_vertices,
        node_coords = node_coords,
        edge_neighbors_indptr = edge_neighbors_indptr,
        edge_neighbors_indices = edge_neighbors_indices,
        vertex_neighbors_indptr = vertex_neighbors_indptr,
        vertex_neighbors_indices = vertex_neighbors_indices,
        tri_index_base = Int64(1),
        node_index_base = Int64(1),
    )
end

"""
    triangleConnectivity(geosInfo)

Flatten `geosInfo` the same way as field/current extraction and compute connectivity.
Requires the flattened geometry to be triangles.
"""
function triangleConnectivity(geosInfo)
    geos_flat = _flatten_geos_basics(geosInfo)
    if !(geos_flat isa AbstractVector{<:TriangleInfo})
        # Fall back to filtering (keeps ordering) but requires all entries be triangles.
        tris = TriangleInfo[]
        for g in geos_flat
            g isa TriangleInfo || error("triangleConnectivity requires TriangleInfo geometries")
            push!(tris, g)
        end
        return triangleConnectivity(tris)
    end
    return triangleConnectivity(geos_flat)
end

FieldData{FT, CT}(npoints::Int, positions::Vector{SVec3D{FT}}) where {FT, CT} = 
    FieldData{FT, CT}(npoints, positions, Dict{Symbol, Vector{SVec3D{CT}}}())

"""
    calIncidentFields(geosInfo, source::ExcitingSource)

Calculate E and H incident fields from `source` at the centroids of geometry elements.
Returns `FieldData`.
"""
function calIncidentFields(geosInfo, source::ExcitingSource)
    # Flatten geometry
    geos_flat = _flatten_geos_basics(geosInfo)
    
    npoints = length(geos_flat)
    FT = Precision.FT
    CT = Complex{FT}
    
    positions = Vector{SVec3D{FT}}(undef, npoints)
    E         = Vector{SVec3D{CT}}(undef, npoints)
    H         = Vector{SVec3D{CT}}(undef, npoints)

    Threads.@threads for i in 1:npoints
        geo = geos_flat[i]
        if hasproperty(geo, :center)
            r = SVec3D{FT}(geo.center)
            positions[i] = r
            E[i] = sourceEfield(source, r)
            H[i] = sourceHfield(source, r)
        end
    end
    
    fd = FieldData{FT, CT}(npoints, positions)
    fd.fields[:E_inc] = E
    fd.fields[:H_inc] = H
    return fd
end

function _flatten_geos_basics(geosInfo::AbstractVector{<:VSCellType})
    return geosInfo
end

function _flatten_geos_basics(geosInfo::AbstractVector{<:AbstractVector})
    return reduce(vcat, geosInfo)
end

function _flatten_geos_basics(geosInfo)
    # Fallback for generic iterables
    geos_flat = []
    for part in geosInfo
        if isa(part, AbstractVector)
             append!(geos_flat, part)
        else
             push!(geos_flat, part)
        end
    end
    return geos_flat
end

# Backward compatibility aliases
calExcitationFields(geosInfo, source) = calIncidentFields(geosInfo, source)


# Internal: evaluate RWG basis value at r on a given triangle with local-id idx_in_geo (1..3)
@inline function _rwg_value_at(r::SVec3D{FT}, bf, tri, idx_in_geo::Int) where {FT}
    sgn = sign(tri.edgel[idx_in_geo])
    l   = bf.edgel
    A   = tri.area
    r_free = SVec3D{FT}(tri.vertices[:, idx_in_geo])
    return (sgn * l / (2A)) * (r - r_free)
end


"""
    mergeFieldData!(target::FieldData, source_raw::NamedTuple)

Merge fields from `source` into `target`. Requires matching number of points.
Note: Does not rigorously check if positions are identical, assumes consistent mesh usage.
"""
function mergeFieldData!(target::FieldData, source_raw::NamedTuple)
    source = toFieldData(source_raw)
    if target.npoints != source.npoints
        error("Cannot merge FieldData: different number of points (target: $(target.npoints), source: $(source.npoints))")
    end
    merge!(target.fields, source.fields)
    return target
end

"""
    saveFieldData(filename::String, data)

Save field data to CSV or NPZ.
"""
function saveFieldData(filename::String, data; connectivity_geosInfo = nothing)
    if endswith(filename, ".npz")
        n = data.npoints
        dict_to_save = Dict{String, Any}()
        
        # Save Positions
        pos_arr = Matrix{eltype(eltype(data.positions))}(undef, n, 3)
        for i in 1:n
            pos_arr[i, 1] = data.positions[i][1]
            pos_arr[i, 2] = data.positions[i][2]
            pos_arr[i, 3] = data.positions[i][3]
        end
        dict_to_save["positions"] = pos_arr
        
        # Save Fields
        for (key, val) in data.fields
             f_arr = Matrix{eltype(eltype(val))}(undef, n, 3)
             for i in 1:n
                 f_arr[i, 1] = val[i][1]
                 f_arr[i, 2] = val[i][2]
                 f_arr[i, 3] = val[i][3]
             end
             dict_to_save[string(key)] = f_arr
        end

        # Optional: Save mesh connectivity in the SAME ordering as `data`.
        if connectivity_geosInfo !== nothing
            conn = triangleConnectivity(connectivity_geosInfo)
            size(conn.tri_vertices, 1) == n || error(
                "Connectivity size mismatch: got $(size(conn.tri_vertices, 1)) triangles but FieldData has $n points."
            )
            dict_to_save["tri_vertices"] = conn.tri_vertices
            dict_to_save["node_coords"] = conn.node_coords
            dict_to_save["tri_neighbors_edge_indptr"] = conn.edge_neighbors_indptr
            dict_to_save["tri_neighbors_edge_indices"] = conn.edge_neighbors_indices
            dict_to_save["tri_neighbors_vertex_indptr"] = conn.vertex_neighbors_indptr
            dict_to_save["tri_neighbors_vertex_indices"] = conn.vertex_neighbors_indices
            dict_to_save["tri_index_base"] = conn.tri_index_base
            dict_to_save["node_index_base"] = conn.node_index_base
        end
        
        npzwrite(filename, dict_to_save)
        
    else
        open(filename, "w") do io
            # Construct Header
            header = "rx,ry,rz"
            sorted_keys = sort(collect(keys(data.fields)))
            for k in sorted_keys
                k_str = string(k)
                header *= ",$(k_str)x_real,$(k_str)x_imag,$(k_str)y_real,$(k_str)y_imag,$(k_str)z_real,$(k_str)z_imag"
            end
            println(io, header)
            
            for i in 1:data.npoints
                r = data.positions[i]
                @printf io "%.6e,%.6e,%.6e" r[1] r[2] r[3]
                
                for k in sorted_keys
                    val = data.fields[k][i]
                    @printf io ",%.6e,%.6e,%.6e,%.6e,%.6e,%.6e" real(val[1]) imag(val[1]) real(val[2]) imag(val[2]) real(val[3]) imag(val[3])
                end
                print(io, "\n")
            end
        end
    end
    nothing
end

# Wrappers for easier usage
function saveIncidentFields(filename::String, geosInfo, source::ExcitingSource)
    data = calIncidentFields(geosInfo, source)
    saveFieldData(filename, data)
end

# Backward compatibility wrappers (can be deprecated later)
saveExcitationFields(filename::String, geosInfo, source::ExcitingSource) = saveIncidentFields(filename, geosInfo, source)
saveExcitationFields(filename::String, data::FieldData) = saveFieldData(filename, data)
saveSurfaceCurrents(filename::String, data::FieldData) = saveFieldData(filename, data)
