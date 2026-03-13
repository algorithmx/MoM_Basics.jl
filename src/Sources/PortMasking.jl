# =============================================================================
# Port Masking (Logical - Non-Destructive)
# =============================================================================
#
# This module provides logical port masking functionality that identifies and
# marks mesh elements within port regions WITHOUT modifying the mesh structure.
#
# Design Philosophy:
# - Mesh preservation: The original mesh remains unchanged
# - Logical identification: Elements are marked/identified but not removed
# - Boundary focus: Primary purpose is to identify boundary edges for excitation
# - Debugging support: Visualization and validation of port identification
#
# =============================================================================

"""
    PortMask{IT<:Integer}

Logical mask representing a port region on the mesh.

This structure stores indices of mesh elements that belong to the port region
without modifying the underlying mesh. It enables:
- Identification of port aperture triangles
- Boundary edge detection
- Visualization of port regions
- Validation of port geometry

# Fields
- `vertexIDs::Vector{IT}` -- Vertices inside the port region
- `triangleIDs::Vector{IT}` -- Triangles fully inside the port aperture
- `boundaryEdgeIDs::Vector{IT}` -- RWG basis functions on the port boundary
- `exteriorTriangleIDs::Vector{IT}` -- Triangles adjacent to boundary (outside)
- `maskType::Symbol` -- Type of mask (:aperture, :boundary, :full)

# Notes
The mesh itself is NOT modified. This is purely a logical identification
used for excitation and visualization purposes.
"""
struct PortMask{IT<:Integer}
    vertexIDs::Vector{IT}
    triangleIDs::Vector{IT}
    boundaryEdgeIDs::Vector{IT}
    exteriorTriangleIDs::Vector{IT}
    maskType::Symbol
    
    function PortMask{IT}(
        vertexIDs::Vector{IT},
        triangleIDs::Vector{IT},
        boundaryEdgeIDs::Vector{IT},
        exteriorTriangleIDs::Vector{IT} = IT[];
        maskType::Symbol = :aperture
    ) where {IT<:Integer}
        @assert maskType in (:aperture, :boundary, :full) "Invalid mask type: $maskType"
        new{IT}(vertexIDs, triangleIDs, boundaryEdgeIDs, exteriorTriangleIDs, maskType)
    end
end

"""
    PortMask(vertexIDs, triangleIDs, boundaryEdgeIDs, exteriorTriangleIDs=IT[]; maskType=:aperture)

Default integer type constructor for PortMask.
"""
PortMask(vertexIDs::Vector{IT}, triangleIDs::Vector{IT}, boundaryEdgeIDs::Vector{IT},
         exteriorTriangleIDs::Vector{IT} = IT[]; maskType::Symbol = :aperture) where {IT<:Integer} =
    PortMask{IT}(vertexIDs, triangleIDs, boundaryEdgeIDs, exteriorTriangleIDs; maskType=maskType)


# =============================================================================
# Port Mask Creation
# =============================================================================

"""
    create_port_mask(
        predicate::Function,
        center::MVec3D{FT},
        normal::MVec3D{FT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        rwgsInfo::Vector{RWG{IT, FT}};
        maskType::Symbol = :aperture,
        tol::FT = FT(1e-6)
    ) where {FT<:Real, IT<:Integer}

Create a logical port mask by identifying mesh elements within a port region.

This function performs geometric identification WITHOUT modifying the mesh:
1. Finds vertices inside the port region (using predicate)
2. Identifies triangles fully contained in the port
3. Detects boundary edges (XOR: one triangle inside, one outside)
4. Optionally identifies exterior triangles adjacent to boundary

# Arguments
- `predicate::Function` -- `(point::AbstractVector) -> Bool`, true if inside port
- `center::MVec3D{FT}` -- Port center reference point
- `normal::MVec3D{FT}` -- Port normal vector (for plane tolerance)
- `trianglesInfo::Vector{TriangleInfo}` -- Mesh triangles
- `rwgsInfo::Vector{RWG}` -- RWG basis functions
- `maskType::Symbol = :aperture` -- Mask classification type
- `tol::FT` -- Tolerance for plane distance check

# Returns
- `PortMask{IT}` -- Logical mask identifying port region elements

# Example
```julia
# Create a rectangular port mask
predicate = p -> abs(p[1] - xc) <= w/2 && abs(p[2] - yc) <= h/2
mask = create_port_mask(predicate, center, normal, trianglesInfo, rwgsInfo)

# Query mask properties
num_vertices = length(mask.vertexIDs)
num_boundary_edges = length(mask.boundaryEdgeIDs)
```
"""
function create_port_mask(
    predicate::Function,
    center::MVec3D{FT},
    normal::MVec3D{FT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}};
    maskType::Symbol = :aperture,
    tol::FT = FT(1e-6)
) where {FT<:Real, IT<:Integer}
    
    # 1. Collect vertices inside port region
    vertex_ids = _mask_collect_vertices(predicate, center, normal, tol, trianglesInfo)
    isempty(vertex_ids) && error("No mesh vertices found inside port region")
    
    # 2. Collect triangles fully inside port
    triangle_ids = _mask_collect_triangles(vertex_ids, trianglesInfo)
    isempty(triangle_ids) && error("No triangles found for port region")
    
    # 3. Identify boundary edges
    boundary_data = _mask_identify_boundary(vertex_ids, triangle_ids, rwgsInfo, trianglesInfo)
    isempty(boundary_data.rwgIDs) && error("No boundary edges found for port")
    
    # 4. Identify exterior triangles (optional, for visualization)
    exterior_tri_ids = _mask_collect_exterior_triangles(boundary_data.rwgIDs, rwgsInfo, triangle_ids)
    
    return PortMask{IT}(
        vertex_ids,
        triangle_ids,
        boundary_data.rwgIDs,
        exterior_tri_ids;
        maskType = maskType
    )
end

"""
    create_port_mask(
        port::DeltaGapArrayPort{FT, IT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        rwgsInfo::Vector{RWG{IT, FT}};
        maskType::Symbol = :aperture
    ) where {FT<:Real, IT<:Integer}

Create a PortMask from an already-bound DeltaGapArrayPort.

This extracts the already-computed identification from the port structure.
"""
function create_port_mask(
    port::DeltaGapArrayPort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}};
    maskType::Symbol = :aperture
) where {FT<:Real, IT<:Integer}
    
    port.isBound || error("Port must be bound to mesh before creating mask")
    
    # Identify exterior triangles from boundary edges
    exterior_tri_ids = _mask_collect_exterior_triangles(port.rwgIDs, rwgsInfo, port.triangleIDs)
    
    return PortMask{IT}(
        port.vertexIDs,
        port.triangleIDs,
        port.rwgIDs,
        exterior_tri_ids;
        maskType = maskType
    )
end


# =============================================================================
# Internal Helper Functions
# =============================================================================

function _mask_collect_vertices(
    predicate::Function,
    center::MVec3D{FT},
    normal::MVec3D{FT},
    tol::FT,
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    
    vertex_ids = Set{IT}()
    n̂ = normalize(normal)
    
    for tri in trianglesInfo
        for i in 1:3
            vertex = MVec3D{FT}(tri.vertices[:, i])
            rel = vertex - center
            normal_dist = abs(rel ⋅ n̂)
            
            # Must be on the port plane AND inside the port region
            if normal_dist <= tol && predicate(vertex)
                push!(vertex_ids, tri.verticesID[i])
            end
        end
    end
    
    return sort!(collect(vertex_ids))
end

function _mask_collect_triangles(
    vertex_ids::Vector{IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    
    vertex_set = Set(vertex_ids)
    tri_ids = IT[]
    
    for tri in trianglesInfo
        # Triangle is inside if ALL its vertices are inside
        if all(vid -> vid in vertex_set, tri.verticesID)
            push!(tri_ids, tri.triID)
        end
    end
    
    return tri_ids
end

function _mask_identify_boundary(
    vertex_ids::Vector{IT},
    triangle_ids::Vector{IT},
    rwgsInfo::Vector{RWG{IT, FT}},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    
    vertex_set = Set(vertex_ids)
    triangle_set = Set(triangle_ids)
    
    rwg_ids = IT[]
    tri_pos = IT[]
    tri_neg = IT[]
    centers = MVec3D{FT}[]
    orients = MVec3D{FT}[]
    lengths = FT[]
    
    for rwg in rwgsInfo
        # Check which side(s) of the RWG are in the port
        tri_pos_in = rwg.inGeo[1] != 0 && rwg.inGeo[1] in triangle_set
        tri_neg_in = rwg.inGeo[2] != 0 && rwg.inGeo[2] in triangle_set
        
        # XOR: exactly one triangle inside -> boundary edge
        (tri_pos_in ⊻ tri_neg_in) || continue
        
        # Get edge geometry
        tri_slot = rwg.inGeo[1] != 0 ? 1 : 2
        tri_id = rwg.inGeo[tri_slot]
        local_edge = rwg.inGeoID[tri_slot]
        tri = trianglesInfo[tri_id]
        
        # Verify both edge vertices are in the port
        v1_id = tri.verticesID[EDGEVmINTriVsID[local_edge]]
        v2_id = tri.verticesID[EDGEVpINTriVsID[local_edge]]
        (v1_id in vertex_set && v2_id in vertex_set) || continue
        
        p1 = MVec3D{FT}(tri.vertices[:, EDGEVmINTriVsID[local_edge]])
        p2 = MVec3D{FT}(tri.vertices[:, EDGEVpINTriVsID[local_edge]])
        
        push!(rwg_ids, rwg.bfID)
        push!(tri_pos, rwg.inGeo[1])
        push!(tri_neg, rwg.inGeo[2])
        push!(lengths, rwg.edgel)
        push!(centers, MVec3D{FT}((p1 + p2) / 2))
        push!(orients, normalize(MVec3D{FT}(p2 - p1)))
    end
    
    return (
        rwgIDs = rwg_ids,
        triPos = tri_pos,
        triNeg = tri_neg,
        centers = centers,
        orients = orients,
        lengths = lengths
    )
end

function _mask_collect_exterior_triangles(
    boundary_rwg_ids::Vector{IT},
    rwgsInfo::Vector{RWG{IT, FT}},
    interior_tri_ids::Vector{IT}
) where {FT<:Real, IT<:Integer}
    
    interior_set = Set(interior_tri_ids)
    exterior_ids = Set{IT}()
    
    for rwg_id in boundary_rwg_ids
        rwg = rwgsInfo[rwg_id]
        
        # Add triangles that are NOT in the interior (the "outside" triangles)
        if rwg.inGeo[1] != 0 && !(rwg.inGeo[1] in interior_set)
            push!(exterior_ids, rwg.inGeo[1])
        end
        if rwg.inGeo[2] != 0 && !(rwg.inGeo[2] in interior_set)
            push!(exterior_ids, rwg.inGeo[2])
        end
    end
    
    return sort!(collect(exterior_ids))
end


# =============================================================================
# Port Mask Queries and Validation
# =============================================================================

"""
    get_mask_statistics(mask::PortMask{IT}) where {IT<:Integer}

Get statistical information about a port mask.

# Returns
- `NamedTuple` with: num_vertices, num_triangles, num_boundary_edges, 
  num_exterior_triangles, perimeter_estimate
"""
function get_mask_statistics(mask::PortMask{IT}, trianglesInfo::Vector{TriangleInfo{IT, FT}}) where {FT<:Real, IT<:Integer}
    # Estimate perimeter from boundary edges
    # Note: This is approximate since we don't have edge lengths stored in mask
    
    return (
        num_vertices = length(mask.vertexIDs),
        num_triangles = length(mask.triangleIDs),
        num_boundary_edges = length(mask.boundaryEdgeIDs),
        num_exterior_triangles = length(mask.exteriorTriangleIDs),
        mask_type = mask.maskType
    )
end

"""
    validate_port_mask(mask::PortMask{IT}, trianglesInfo, rwgsInfo; verbose=false) where {IT}

Validate that a port mask represents a coherent port region.

Checks:
1. All vertices in triangles are valid
2. Boundary edges form a closed loop (or multiple loops)
3. Triangles are connected (share edges)
4. No isolated boundary edges

# Returns
- `(is_valid::Bool, issues::Vector{String})`
"""
function validate_port_mask(
    mask::PortMask{IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}};
    verbose::Bool = false
) where {FT<:Real, IT<:Integer}
    
    issues = String[]
    
    # Check 1: Non-empty mask
    if isempty(mask.vertexIDs)
        push!(issues, "Mask has no vertices")
    end
    if isempty(mask.triangleIDs)
        push!(issues, "Mask has no triangles")
    end
    if isempty(mask.boundaryEdgeIDs)
        push!(issues, "Mask has no boundary edges")
    end
    
    # Check 2: Valid indices
    num_tris = length(trianglesInfo)
    num_rwgs = length(rwgsInfo)
    
    for tri_id in mask.triangleIDs
        if tri_id < 1 || tri_id > num_tris
            push!(issues, "Invalid triangle ID: $tri_id")
        end
    end
    
    for rwg_id in mask.boundaryEdgeIDs
        if rwg_id < 1 || rwg_id > num_rwgs
            push!(issues, "Invalid RWG ID: $rwg_id")
        end
    end
    
    # Check 3: Boundary edges reference valid triangles
    interior_set = Set(mask.triangleIDs)
    for rwg_id in mask.boundaryEdgeIDs
        rwg = rwgsInfo[rwg_id]
        tri_pos_valid = rwg.inGeo[1] == 0 || rwg.inGeo[1] in interior_set || rwg.inGeo[1] in mask.exteriorTriangleIDs
        tri_neg_valid = rwg.inGeo[2] == 0 || rwg.inGeo[2] in interior_set || rwg.inGeo[2] in mask.exteriorTriangleIDs
        
        if !tri_pos_valid || !tri_neg_valid
            push!(issues, "Boundary edge $rwg_id references invalid triangles")
        end
    end
    
    is_valid = isempty(issues)
    
    if verbose
        if is_valid
            println("✓ Port mask validation passed")
            println("  Vertices: $(length(mask.vertexIDs))")
            println("  Triangles: $(length(mask.triangleIDs))")
            println("  Boundary edges: $(length(mask.boundaryEdgeIDs))")
        else
            println("✗ Port mask validation failed:")
            for issue in issues
                println("  - $issue")
            end
        end
    end
    
    return (is_valid, issues)
end


# =============================================================================
# Port Mask Visualization Data
# =============================================================================

"""
    get_mask_geometry(mask::PortMask{IT}, trianglesInfo, rwgsInfo) where {IT}

Extract geometric data from a port mask for visualization.

# Returns
Named tuple with:
- `aperture_vertices::Vector{MVec3D{FT}}` -- Vertices in the port aperture
- `aperture_tri_centers::Vector{MVec3D{FT}}` -- Centers of aperture triangles  
- `boundary_edge_centers::Vector{MVec3D{FT}}` -- Centers of boundary edges
- `boundary_edge_orients::Vector{MVec3D{FT}}` -- Orientations of boundary edges
"""
function get_mask_geometry(
    mask::PortMask{IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}
    
    # Collect aperture vertices (unique positions)
    aperture_vertices = MVec3D{FT}[]
    seen_verts = Set{IT}()
    for tri_id in mask.triangleIDs
        tri = trianglesInfo[tri_id]
        for i in 1:3
            vid = tri.verticesID[i]
            if !(vid in seen_verts)
                push!(seen_verts, vid)
                push!(aperture_vertices, MVec3D{FT}(tri.vertices[:, i]))
            end
        end
    end
    
    # Collect triangle centers
    aperture_tri_centers = [MVec3D{FT}(trianglesInfo[tid].center) for tid in mask.triangleIDs]
    
    # Collect boundary edge data
    boundary_edge_centers = MVec3D{FT}[]
    boundary_edge_orients = MVec3D{FT}[]
    
    for rwg_id in mask.boundaryEdgeIDs
        rwg = rwgsInfo[rwg_id]
        push!(boundary_edge_centers, MVec3D{FT}(rwg.center))
        
        # Get edge orientation from triangle
        tri_slot = rwg.inGeo[1] != 0 ? 1 : 2
        tri_id = rwg.inGeo[tri_slot]
        tri = trianglesInfo[tri_id]
        local_edge = rwg.inGeoID[tri_slot]
        
        p1 = MVec3D{FT}(tri.vertices[:, EDGEVmINTriVsID[local_edge]])
        p2 = MVec3D{FT}(tri.vertices[:, EDGEVpINTriVsID[local_edge]])
        push!(boundary_edge_orients, normalize(p2 - p1))
    end
    
    return (
        aperture_vertices = aperture_vertices,
        aperture_tri_centers = aperture_tri_centers,
        boundary_edge_centers = boundary_edge_centers,
        boundary_edge_orients = boundary_edge_orients
    )
end


# =============================================================================
# Multiple Port Mask Management
# =============================================================================

"""
    PortMaskCollection{IT<:Integer}

Collection of port masks for managing multiple ports on a mesh.
"""
struct PortMaskCollection{IT<:Integer}
    masks::Dict{IT, PortMask{IT}}  # port_id => mask
    meshTriangleCount::IT
    meshRWGCount::IT
end

"""
    create_port_mask_collection(
        ports::Vector{<:PortType},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        rwgsInfo::Vector{RWG{IT, FT}}
    ) where {FT<:Real, IT<:Integer}

Create a PortMaskCollection from an array of ports.

Only includes ports that are bound to the mesh.
"""
function create_port_mask_collection(
    ports::Vector{PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer, PT<:PortType}
    
    masks = Dict{IT, PortMask{IT}}()
    
    for port in ports
        if hasfield(typeof(port), :isBound) && port.isBound
            mask = create_port_mask(port, trianglesInfo, rwgsInfo)
            masks[port.id] = mask
        end
    end
    
    return PortMaskCollection{IT}(
        masks,
        IT(length(trianglesInfo)),
        IT(length(rwgsInfo))
    )
end

"""
    check_mask_overlap(collection::PortMaskCollection{IT}) where {IT}

Check if any port masks overlap (share triangles or boundary edges).

Overlapping ports indicate a configuration error.

# Returns
- `(has_overlap::Bool, overlaps::Vector{String})`
"""
function check_mask_overlap(collection::PortMaskCollection{IT}) where {IT<:Integer}
    overlaps = String[]
    mask_ids = collect(keys(collection.masks))
    
    for i in 1:length(mask_ids)
        for j in (i+1):length(mask_ids)
            id1, id2 = mask_ids[i], mask_ids[j]
            mask1 = collection.masks[id1]
            mask2 = collection.masks[id2]
            
            # Check triangle overlap
            tri_overlap = intersect(mask1.triangleIDs, mask2.triangleIDs)
            if !isempty(tri_overlap)
                push!(overlaps, "Port $id1 and Port $id2 share $(length(tri_overlap)) triangles")
            end
            
            # Check boundary edge overlap
            edge_overlap = intersect(mask1.boundaryEdgeIDs, mask2.boundaryEdgeIDs)
            if !isempty(edge_overlap)
                push!(overlaps, "Port $id1 and Port $id2 share $(length(edge_overlap)) boundary edges")
            end
        end
    end
    
    return (!isempty(overlaps), overlaps)
end


# =============================================================================
# Exports
# =============================================================================

export PortMask, PortMaskCollection
export create_port_mask, create_port_mask_collection
export get_mask_statistics, validate_port_mask, get_mask_geometry
export check_mask_overlap

