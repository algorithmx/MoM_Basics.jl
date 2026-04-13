"""
    DeltaGapArrayPort{FT<:Real, IT<:Integer, DT<:AbstractExcitationDistribution{FT}} <: PortType

Delta-Gap array port excitation for arbitrary cross-section geometries.

This port type applies excitation at the **port boundary edges** (perimeter) using
an array of Delta-Gap voltage sources. Each RWG basis function on the port 
boundary receives a voltage excitation weighted by the configured distribution
(e.g., uniform or single-side distribution). This generalizes port excitation to arbitrary
cross-section shapes while preserving the essential features of `DeltaGapPort`.

# Excitation Mechanism

The port identifies mesh edges on the port boundary (where one adjacent triangle
is inside the port region and one is outside) and applies Delta-Gap excitation:

```
V_excitation[rwgID] = V_port × weight(edge_center) × edge_length / 2
```

where the weight is determined by the `excitationDistribution`.
For half-RWGs (boundary edges), the full edge length is used.

# Design Philosophy

1. **Separation of Concerns**: Geometry specification, mesh binding, and excitation 
   computation are separate phases.
2. **Lazy Evaluation**: Mesh-dependent properties (edges, weights) are computed on demand 
   or via explicit `bind_to_mesh!` call.
3. **Generic Core**: Common functionality (S-parameters, excitation vectors) is implemented
   at the abstract level using a minimal interface.
4. **Extensible Geometry**: New cross-section shapes can be added by defining the 
   `PortGeometry` interface without modifying the core port type.

# Fields

## Identification & Excitation
- `id::IT` -- Port identifier
- `V::Complex{FT}` -- Reference excitation voltage
- `freq::FT` -- Operating frequency (Hz)
- `portType::Symbol` -- Always `:delta_gap_array`
- `excitationDistribution::DT` -- Voltage distribution pattern (uniform, single-side, etc.)
- `modeImpedance::Complex{FT}` -- Characteristic impedance (computed from distribution)

## Geometry (Shape-Agnostic)
- `center::MVec3D{FT}` -- Port center (reference point for mode calculations)
- `normal::MVec3D{FT}` -- Propagation direction (unit vector)
- `widthDir::MVec3D{FT}` -- Reference direction for "width" (mode coordinate ξ)
- `heightDir::MVec3D{FT}` -- Reference direction for "height" (mode coordinate η)

## Mesh Binding (Computed)
- `isBound::Bool` -- Whether port has been bound to mesh
- `vertexIDs::Vector{IT}` -- Mesh vertices in port region
- `triangleIDs::Vector{IT}` -- Triangles in port region
- `rwgIDs::Vector{IT}` -- RWG basis functions on port boundary
- `triID_pos::Vector{IT}` -- Positive triangle IDs for each RWG
- `triID_neg::Vector{IT}` -- Negative triangle IDs for each RWG (0 = aperture)
- `edgeLengths::Vector{FT}` -- Lengths of boundary edges
- `edgeCenters::Vector{MVec3D{FT}}` -- Centers of boundary edges
- `edgeOrient::Vector{MVec3D{FT}}` -- Orientation vectors of edges
- `edgeWeights::Vector{Complex{FT}}` -- Voltage weights from distribution

## Excitation Mode
- `singleEdgeMode::Bool` -- Use single-edge excitation (like DeltaGapPort)
- `primaryRwgID::IT` -- Primary RWG for single-edge mode

## Status
- `isActive::Bool` -- Whether port is active

# Usage Pattern

```julia
# 1. Create port with geometry (shape-agnostic at this point)
port = DeltaGapArrayPort(;
    center = [0, 0, 0],
    normal = [0, 0, 1],
    widthDir = [1, 0, 0],  # Optional: auto-computed if not provided
    excitationDistribution = UniformDistribution(),
    freq = 10e9
)

# 2. Bind to mesh (shape-specific logic applied here)
bind_to_mesh!(port, trianglesInfo, rwgsInfo) do point
    # Shape predicate: returns true if point is inside port cross-section
    abs(point[1]) <= width/2 && abs(point[2]) <= height/2
end

# 3. Use port (generic methods work with any shape)
V_exc = excitationVectorEFIE(port, trianglesInfo, nbf)
Z_in = computeInputImpedance(port, Z_matrix, V_exc)
S11 = computeS11(port, Z_matrix, V_exc)
```

# Extending with New Shapes

To add a new cross-section shape (e.g., elliptical), you only need to:
1. Define a predicate function `(point) -> Bool`
2. Optionally define `estimate_port_dimensions()` for mode impedance
3. Use `bind_to_mesh!(port, predicate, trianglesInfo, rwgsInfo)`
"""
mutable struct DeltaGapArrayPort{
    FT<:Real,
    IT<:Integer,
    DT<:AbstractExcitationDistribution{FT}
} <: PortType
    # Identification & Excitation
    id::IT
    V::Complex{FT}
    freq::FT
    portType::Symbol
    excitationDistribution::DT
    modeImpedance::Complex{FT}
    
    # Geometry (Coordinate Frame)
    center::MVec3D{FT}
    normal::MVec3D{FT}
    widthDir::MVec3D{FT}
    heightDir::MVec3D{FT}
    
    # Mesh Binding State
    isBound::Bool
    vertexIDs::Vector{IT}
    triangleIDs::Vector{IT}
    rwgIDs::Vector{IT}
    triID_pos::Vector{IT}
    triID_neg::Vector{IT}
    edgeLengths::Vector{FT}
    edgeCenters::Vector{MVec3D{FT}}
    edgeOrient::Vector{MVec3D{FT}}
    edgeWeights::Vector{Complex{FT}}
    
    # Excitation Mode
    singleEdgeMode::Bool
    primaryRwgID::IT
    
    # Status
    isActive::Bool
end


# =============================================================================
# Minimal Constructor (Shape-Agnostic)
# =============================================================================

"""
    DeltaGapArrayPort{FT, IT}(; V=1, kwargs...)

Create a DeltaGapArrayPort without binding to mesh.

The port is created in an "unbound" state. Call `bind_to_mesh!` to associate
it with a specific mesh and cross-section shape.

# Arguments
- `id::IT = 0` -- Port identifier
- `V::Number = 1` -- Excitation voltage (Real or Complex). 
  Automatically converted to Complex{FT}. Defaults to 1.0.
- `freq::FT = 0` -- Operating frequency
- `center::MVec3D{FT}` -- Port center position (required)
- `normal::MVec3D{FT}` -- Port normal vector (required)
- `widthDir::MVec3D{FT}` -- Width reference direction (optional, auto-computed)
- `excitationDistribution` -- Voltage distribution pattern (default: UniformDistribution())
- `isActive::Bool = true` -- Port status

The voltage can be specified as real (e.g., `V=1.0`) or complex (e.g., `V=1.0+0.5im`).
Real values are automatically converted to complex with zero imaginary part.

# Example
```julia
port = DeltaGapArrayPort(;
    center = [0, 0, 0],
    normal = [0, 0, 1],
    excitationDistribution = UniformDistribution(),
    freq = 10e9
)

# Later, bind to mesh with specific shape
bind_to_mesh!(port, trianglesInfo, rwgsInfo) do p
    abs(p[1]) <= 0.01 && abs(p[2]) <= 0.005  # Rectangular
end
```
"""
function DeltaGapArrayPort{FT, IT}(;
    id::IT = zero(IT),
    V::Number = 1,  # Accept Real or Complex, default to 1
    freq::FT = zero(FT),
    center::MVec3D{FT},
    normal::MVec3D{FT},
    widthDir::MVec3D{FT} = zero(MVec3D{FT}),
    excitationDistribution::AbstractExcitationDistribution{FT} = UniformDistribution(),
    isActive::Bool = true
) where {FT<:Real, IT<:Integer}
    
    # Convert V to Complex{FT} (handles both Real and Complex inputs)
    V_complex = convert(Complex{FT}, complex(V))
    
    # Compute orthonormal frame
    n̂, wdir, hdir = _compute_port_frame(normal, widthDir)
    
    # Mode impedance (will be Inf for non-modal distributions)
    mode_impedance = _compute_mode_impedance_generic(excitationDistribution, freq, FT(0), FT(0))
    
    return DeltaGapArrayPort{FT, IT, typeof(excitationDistribution)}(
        id, V_complex, freq, :delta_gap_array, excitationDistribution, mode_impedance,
        center, n̂, wdir, hdir,
        false,  # isBound
        IT[], IT[], IT[], IT[], IT[],  # mesh arrays empty
        FT[], MVec3D{FT}[], MVec3D{FT}[], Complex{FT}[],  # edge arrays empty
        false, zero(IT),  # singleEdgeMode, primaryRwgID
        isActive
    )
end

"""
    DeltaGapArrayPort(args...; kwargs...)

Default precision constructor.
"""
DeltaGapArrayPort(args...; kwargs...) = 
    DeltaGapArrayPort{Precision.FT, IntDtype}(args...; kwargs...)


# =============================================================================
# Mesh Binding (Shape-Specific Computation)
# =============================================================================

"""
    bind_to_mesh!(port::DeltaGapArrayPort, predicate, trianglesInfo, rwgsInfo; 
                  estimateDimensions=true)

Bind an unbound port to a mesh, identifying boundary edges for Delta-Gap excitation.

This function:
1. Identifies mesh vertices inside the port region using `predicate`
2. Finds triangles completely inside the port
3. **Identifies boundary edges** (RWG basis functions where exactly one adjacent 
   triangle is inside the port - these form the port perimeter)
4. Computes voltage weights for each boundary edge based on `excitationDistribution`

The boundary edges are where Delta-Gap excitations will be applied during 
`excitationVectorEFIE!`.

# Arguments
- `port::DeltaGapArrayPort` -- The port to bind
- `predicate::Function` -- `(point::AbstractVector) -> Bool`, true if inside port
- `trianglesInfo::Vector{TriangleInfo}` -- Mesh triangles
- `rwgsInfo::Vector{RWG}` -- RWG basis functions
- `estimateDimensions::Bool = true` -- Estimate width/height from edge bounding box

# Example
```julia
# Rectangular port
bind_to_mesh!(port, trianglesInfo, rwgsInfo) do p
    abs(p[1] - xc) <= width/2 && abs(p[2] - yc) <= height/2
end

# Circular port  
bind_to_mesh!(port, trianglesInfo, rwgsInfo) do p
    (p[1] - xc)^2 + (p[2] - yc)^2 <= radius^2
end
```

See also: [`find_port_boundary_edges`](@ref), [`_compute_edge_weights`](@ref)
"""

"""
    find_port_boundary_edges(predicate, port_center, port_normal, trianglesInfo, rwgsInfo)

Find boundary edges for a port defined by a predicate function.

This is a low-level function that performs edge discovery without computing weights.
Returns a named tuple with all edge geometry information.

# Arguments
- `predicate::Function` -- `(point::AbstractVector) -> Bool`, true if inside port
- `port_center::MVec3D` -- Port center for tolerance computation
- `port_normal::MVec3D` -- Port normal for tolerance computation
- `trianglesInfo::Vector{TriangleInfo}` -- Mesh triangles
- `rwgsInfo::Vector{RWG}` -- RWG basis functions

# Returns
Named tuple `(rwgIDs, triPos, triNeg, lengths, centers, orients)` with boundary edge data.
"""
function find_port_boundary_edges(
    predicate::Function,
    port_center::MVec3D{FT},
    port_normal::MVec3D{FT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}
    
    # 1. Collect vertices inside port region
    tol = _compute_tolerance(trianglesInfo)
    vertex_ids = _collect_vertices_by_predicate(predicate, port_center, port_normal, 
                                                  tol, trianglesInfo)
    isempty(vertex_ids) && error("No mesh vertices found inside port region")
    
    # 2. Collect triangles (all vertices inside)
    triangle_ids = _collect_triangles_by_vertices(vertex_ids, trianglesInfo)
    isempty(triangle_ids) && error("No triangles found for port region")
    
    # 3. Identify boundary edges (XOR: one tri inside, one outside)
    boundary_data = _identify_boundary_edges(vertex_ids, triangle_ids, rwgsInfo, trianglesInfo)
    isempty(boundary_data) && error("No boundary edges found")
    
    return boundary_data
end


"""
    bind_to_mesh!(
        port::DeltaGapArrayPort, predicate, trianglesInfo, rwgsInfo; 
        estimateDimensions=true, forcePortPEC=true
    )

Bind an unbound port to a mesh, identifying boundary edges for Delta-Gap excitation.

This convenience function combines edge discovery and weight computation for generic
ports. For rectangular ports with known dimensions, consider using the lower-level
`find_port_boundary_edges` + `_compute_edge_weights(centers, port, width, height)` directly.

# Keyword Arguments
- `estimateDimensions::Bool = true` -- Estimate width/height from edge bounding box for mode impedance
- `forcePortPEC::Bool = true` -- Set surface impedance `Zs = 0` (PEC) on port triangles.
  For metallic waveguide ports, the port surface should remain PEC even if the surrounding
  structure has non-PEC materials. Set to `false` only if you explicitly want the port
  to inherit the material properties of the underlying mesh.

See also: [`find_port_boundary_edges`](@ref), [`_compute_edge_weights`](@ref)
"""
function bind_to_mesh!(
    port::DeltaGapArrayPort{FT, IT, DT},
    predicate::Function,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}};
    estimateDimensions::Bool = true,
    forcePortPEC::Bool = true
) where {FT<:Real, IT<:Integer, DT}
    
    port.isBound && @warn "Port $(port.id) is already bound. Rebinding..."
    
    # 1. Collect vertices inside port region (needed for updating port fields)
    tol = _compute_tolerance(trianglesInfo)
    vertex_ids = _collect_vertices_by_predicate(predicate, port.center, port.normal, 
                                                  tol, trianglesInfo)
    isempty(vertex_ids) && error("No mesh vertices found inside port region")
    
    # 2. Collect triangles (all vertices inside)
    triangle_ids = _collect_triangles_by_vertices(vertex_ids, trianglesInfo)
    isempty(triangle_ids) && error("No triangles found for port region")
    
    # 3. Enforce PEC on port surface triangles (for metallic waveguide ports)
    # This ensures the port aperture remains PEC even if surrounding structure has IBC
    if forcePortPEC
        for tri_id in triangle_ids
            trianglesInfo[tri_id].Zs = zero(Complex{FT})
        end
    end
    
    # 4. Find boundary edges (geometry only)
    boundary_data = _identify_boundary_edges(vertex_ids, triangle_ids, rwgsInfo, trianglesInfo)
    isempty(boundary_data) && error("No boundary edges found")
    
    # 5. Compute voltage weights (estimate dimensions from actual edges)
    edge_weights = _compute_edge_weights(boundary_data.centers, port)
    
    # 6. Update port fields
    port.vertexIDs = vertex_ids
    port.triangleIDs = triangle_ids
    port.rwgIDs = boundary_data.rwgIDs
    port.triID_pos = boundary_data.triPos
    port.triID_neg = boundary_data.triNeg
    port.edgeLengths = boundary_data.lengths
    port.edgeCenters = boundary_data.centers
    port.edgeOrient = boundary_data.orients
    port.edgeWeights = edge_weights
    port.isBound = true
    
    # 7. Optionally estimate dimensions for mode impedance
    if estimateDimensions
        width, height = _estimate_dimensions_from_edges(port.edgeCenters, port.center,
                                                         port.widthDir, port.heightDir)
        port.modeImpedance = _compute_mode_impedance_generic(
            port.excitationDistribution, port.freq, width, height
        )
    end
    
    return port
end

"""
    unbind_mesh!(port::DeltaGapArrayPort)

Remove mesh binding, returning port to unbound state.
"""
function unbind_mesh!(port::DeltaGapArrayPort{FT, IT, DT}) where {FT<:Real, IT<:Integer, DT}
    port.isBound = false
    port.vertexIDs = IT[]
    port.triangleIDs = IT[]
    port.rwgIDs = IT[]
    port.triID_pos = IT[]
    port.triID_neg = IT[]
    port.edgeLengths = FT[]
    port.edgeCenters = MVec3D{FT}[]
    port.edgeOrient = MVec3D{FT}[]
    port.edgeWeights = Complex{FT}[]
    return port
end


# =============================================================================
# Generic Excitation Methods (Work with Any Shape)
# =============================================================================

function sourceEfield(port::DeltaGapArrayPort{FT, IT, DT}, r::AbstractVector{FT}) where {FT, IT, DT}
    return zero(MVec3D{Complex{FT}})
end

function sourceHfield(port::DeltaGapArrayPort{FT, IT, DT}, r::AbstractVector{FT}) where {FT, IT, DT}
    return zero(MVec3D{Complex{FT}})
end


# Excitation vector functions are implemented in MoM_Kernels.jl (SurfacePortExcitation.jl)


# =============================================================================
# Generic S-Parameter Methods (Work with Any Shape)
# =============================================================================
# NOTE: computeInputImpedance, computeS11 are now defined
# in MoM_Kernels.jl for all port types to be consistent with the architecture.
# They are kept in MoM_Basics as well for backward compatibility with the
# DeltaGapArrayPort specific implementation.

function getPortVoltage(port::DeltaGapArrayPort{FT, IT, DT}, current::Complex{FT}) where {FT, IT, DT}
    return port.V
end

function getPortCurrent(port::DeltaGapArrayPort{FT, IT, DT}; Z::Complex{FT} = Complex{FT}(50.0)) where {FT, IT, DT}
    return port.V / Z
end


# =============================================================================
# Configuration Methods
# =============================================================================

"""
    set_excitation_mode!(port, mode::Symbol; rwgID=0)

Set excitation mode: `:array` (default) or `:single_edge`.
"""
function set_excitation_mode!(
    port::DeltaGapArrayPort{FT, IT, DT},
    mode::Symbol;
    rwgID::IT = zero(IT)
) where {FT, IT, DT}
    if mode == :array
        port.singleEdgeMode = false
        port.primaryRwgID = zero(IT)
    elseif mode == :single_edge
        port.singleEdgeMode = true
        if port.isBound
            idx = findfirst(==(rwgID), port.rwgIDs)
            port.primaryRwgID = idx !== nothing ? rwgID : 
                                (isempty(port.rwgIDs) ? zero(IT) : first(port.rwgIDs))
        else
            port.primaryRwgID = rwgID
        end
    else
        error("Unknown mode: $mode. Use :array or :single_edge")
    end
    return port
end

"""
    set_excitation_distribution!(port, distribution)

Change the excitation distribution. Recomputes edge weights if bound.

For `DeltaGapArrayPort`, dimensions are estimated from edge positions. 
For `RectangularEdgePort`, use the specialized method that uses stored dimensions.
"""
function set_excitation_distribution!(
    port::DeltaGapArrayPort{FT, IT, DT},
    distribution::AbstractExcitationDistribution{FT}
) where {FT, IT, DT}
    port.excitationDistribution = distribution
    
    if port.isBound
        # Use estimated dimensions (no explicit dimensions stored in DeltaGapArrayPort)
        port.edgeWeights = _compute_edge_weights(port.edgeCenters, port)
    end
    
    # Update mode impedance using estimated dimensions
    width, height = port.isBound ? 
        _estimate_dimensions_from_edges(port.edgeCenters, port.center, port.widthDir, port.heightDir) :
        (zero(FT), zero(FT))
    port.modeImpedance = _compute_mode_impedance_generic(distribution, port.freq, width, height)
    
    return port
end


# =============================================================================
# Internal Helper Functions (Private)
# =============================================================================

function _compute_port_frame(normal::MVec3D{FT}, widthDir::MVec3D{FT}) where {FT}
    n̂ = _normalize_port_vector(normal)
    wdir = widthDir
    if iszero(wdir)
        ref = abs(n̂[1]) < FT(0.9) ? MVec3D{FT}(1, 0, 0) : MVec3D{FT}(0, 1, 0)
        wdir = ref - (ref ⋅ n̂) * n̂
    else
        wdir = wdir - (wdir ⋅ n̂) * n̂
    end
    wdir = _normalize_port_vector(wdir)
    hdir = _normalize_port_vector(cross(n̂, wdir))
    return n̂, wdir, hdir
end

function _compute_mode_impedance_generic(dist::AbstractExcitationDistribution{FT}, f::FT, a::FT, b::FT) where {FT}
    return Complex{FT}(Inf)  # Mode impedance not defined for generic distributions
end

function _compute_tolerance(trianglesInfo::Vector{TriangleInfo{IT, FT}}) where {IT, FT}
    # Estimate tolerance from average triangle size
    isempty(trianglesInfo) && return FT(1e-6)
    avg_edge = sqrt(sum(tri.area for tri in trianglesInfo) / length(trianglesInfo))
    return avg_edge * FT(1e-3)
end

"""
    _collect_vertices_by_predicate(predicate, center, normal, tol, trianglesInfo)

Collect the unique mesh vertex IDs that belong to the port cross-section candidate set.

For every triangle in `trianglesInfo`, this helper inspects its three vertices and keeps a
vertex when both of the following conditions are satisfied:

1. The vertex lies close to the port plane, i.e.
     `abs((vertex - center) ⋅ normal) <= tol`
2. `predicate(vertex)` returns `true`

The returned vector contains unique vertex IDs sorted in ascending order.

# Arguments
- `predicate` -- Function receiving a vertex position `MVec3D{FT}` and returning `true`
    when that point is inside the intended port cross-section in the port plane.
- `center` -- Reference point on the port plane.
- `normal` -- Port-plane normal vector used to reject vertices that are too far away from
    the plane.
- `tol` -- Distance tolerance around the plane. This converts the ideal plane test into a
    finite-thickness slab test to accommodate mesh/discretization error.
- `trianglesInfo` -- Mesh triangles that provide both vertex coordinates and vertex IDs.

# Returns
- `Vector{IT}` -- Sorted unique vertex IDs passing the plane-distance and predicate tests.

# Notes
- This function performs a **vertex-level** filter only. It does not decide whether an
    entire triangle belongs to the port region; that is handled later by
    `_collect_triangles_by_vertices`.
- `trianglesInfo` is assumed to be non-empty because element types are inferred from the
    first triangle.
"""
function _collect_vertices_by_predicate(
    predicate::Function,
    center::MVec3D{FT},
    normal::MVec3D{FT},
    tol::FT,
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    vertex_ids = Set{IT}()
    
    for tri in trianglesInfo
        for i in 1:3
            vertex = MVec3D{FT}(tri.vertices[:, i])
            rel = vertex - center
            normal_dist = abs(rel ⋅ normal)
            if normal_dist <= tol && predicate(vertex)
                push!(vertex_ids, tri.verticesID[i])
            end
        end
    end
    return sort!(collect(vertex_ids))
end

function _collect_triangles_by_vertices(vertex_ids, trianglesInfo)
    IT = eltype(trianglesInfo[1].verticesID)
    vertex_set = Set(vertex_ids)
    tri_ids = IT[]
    
    for tri in trianglesInfo
        if all(vid -> vid in vertex_set, tri.verticesID)
            push!(tri_ids, tri.triID)
        end
    end
    return tri_ids
end

"""
    _identify_boundary_edges(
        vertex_ids::Vector{IT},
        triangle_ids::Vector{IT},
        rwgsInfo::Vector{RWG{IT, FT}},
        trianglesInfo::Vector{TriangleInfo{IT, FT}}
    ) where {IT<:Integer, FT<:AbstractFloat}

Identify RWG basis functions on the port boundary (perimeter edges).

# Algorithm Description

## Step 1: XOR Condition (Port Boundary Detection)
Each RWG basis function spans two adjacent triangles: a "positive" triangle (`inGeo[1]`)
and a "negative" triangle (`inGeo[2]`). A basis function lies on the port boundary if 
**exactly one** of its adjacent triangles is inside the port region (XOR condition):

```
Boundary RWG ⇔ (tri⁺ ∈ port) ⊻ (tri⁻ ∈ port)
```

- If both triangles are inside: internal edge (not on boundary)
- If both triangles are outside: external edge (not on boundary)  
- If exactly one is inside: boundary edge (perimeter of port region)

## Step 2: Vertex Verification
After identifying candidate boundary RWGs via XOR, we verify that **both vertices** 
of the edge lie within the port vertex set. This ensures the edge is truly part of 
the port perimeter and handles edge cases where triangles straddle the boundary.

## Step 3: Geometry Extraction
For each verified boundary edge, extract:
- `bfID`: RWG basis function ID (for excitation)
- `inGeo`: Positive and negative triangle IDs (for field evaluation)
- `edgel`: Edge length
- Edge center and orientation vectors (for voltage weight computation)

# Arguments
- `vertex_ids::Vector{IT}` -- IDs of vertices inside the port region
- `triangle_ids::Vector{IT}` -- IDs of triangles fully inside the port region
- `rwgsInfo::Vector{RWG{IT, FT}}` -- All RWG basis functions in the mesh
- `trianglesInfo::Vector{TriangleInfo{IT, FT}}` -- All triangles in the mesh

# Returns
Named tuple `(rwgIDs, triPos, triNeg, lengths, centers, orients)` where:
- `rwgIDs::Vector{IT}` -- RWG basis function IDs on port boundary
- `triPos::Vector{IT}` -- Positive triangle IDs for each boundary RWG (0 = aperture)
- `triNeg::Vector{IT}` -- Negative triangle IDs for each boundary RWG (0 = aperture)
- `lengths::Vector{FT}` -- Edge lengths
- `centers::Vector{MVec3D{FT}}` -- Edge center positions
- `orients::Vector{MVec3D{FT}}` -- Unit vectors along edge direction

# Physical Interpretation

The boundary edges form a **closed contour** (perimeter) around the port region.
Each boundary RWG represents a half-basis function where:
- One triangle is inside the port (carries current)
- The other is outside the port (current terminates at boundary)

These edges are where Delta-Gap voltage sources are applied to excite the port.

# See Also
- [`bind_to_mesh!`](@ref) -- Uses this function to identify excitation edges
- [`_collect_triangles_by_vertices`](@ref) -- Identifies triangles from vertices
"""
function _identify_boundary_edges(
    vertex_ids::Vector{IT},
    triangle_ids::Vector{IT},
    rwgsInfo::Vector{RWG{IT, FT}},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {IT<:Integer, FT<:AbstractFloat}
    
    vertex_set = Set(vertex_ids)
    triangle_set = Set(triangle_ids)
    
    rwg_ids = IT[]
    tri_pos = IT[]
    tri_neg = IT[]
    lengths = FT[]
    centers = MVec3D{FT}[]
    orients = MVec3D{FT}[]
    
    for rwg in rwgsInfo
        # Step 1: XOR condition - exactly one triangle is inside port
        tri_in_pos = rwg.inGeo[1] != 0 && rwg.inGeo[1] in triangle_set
        tri_in_neg = rwg.inGeo[2] != 0 && rwg.inGeo[2] in triangle_set
        (tri_in_pos ⊻ tri_in_neg) || continue
        
        # Step 2: Get edge geometry from the triangle inside the port
        tri_slot = rwg.inGeo[1] != 0 ? 1 : 2
        tri_id = rwg.inGeo[tri_slot]
        local_edge = rwg.inGeoID[tri_slot]
        tri = trianglesInfo[tri_id]
        
        # Step 3: Verify both edge vertices are in port vertex set
        v1_id = tri.verticesID[EDGEVmINTriVsID[local_edge]]
        v2_id = tri.verticesID[EDGEVpINTriVsID[local_edge]]
        (v1_id in vertex_set && v2_id in vertex_set) || continue
        
        # Step 4: Extract edge geometry
        p1 = MVec3D{FT}(tri.vertices[:, EDGEVmINTriVsID[local_edge]])
        p2 = MVec3D{FT}(tri.vertices[:, EDGEVpINTriVsID[local_edge]])
        
        push!(rwg_ids, rwg.bfID)
        push!(tri_pos, rwg.inGeo[1])
        push!(tri_neg, rwg.inGeo[2])
        push!(lengths, rwg.edgel)
        push!(centers, MVec3D{FT}((p1 + p2) / 2))
        push!(orients, _normalize_port_vector(p2 - p1))
    end
    
    return (rwgIDs=rwg_ids, triPos=tri_pos, triNeg=tri_neg, 
            lengths=lengths, centers=centers, orients=orients)
end

"""
    _compute_edge_weights(centers, port::DeltaGapArrayPort, width, height)

Compute edge weights using EXPLICIT dimensions (for rectangular ports with known geometry).

# Arguments
- `centers::Vector{MVec3D}` -- Edge center positions
- `port::DeltaGapArrayPort` -- Port (for distribution, coordinate frame)
- `width::Real` -- Known port width
- `height::Real` -- Known port height

This is the PREFERRED method for `RectangularEdgePort` which has exact dimensions.
"""
function _compute_edge_weights(
    centers::Vector{MVec3D{FT}}, 
    port::DeltaGapArrayPort{FT, IT, DT},
    width::FT,
    height::FT
) where {FT, IT, DT}
    port_params = (
        center = port.center,
        widthDir = port.widthDir,
        heightDir = port.heightDir,
        normal = port.normal,
        width = width,
        height = height
    )
    return [compute_voltage(port.excitationDistribution, c, port_params) for c in centers]
end

"""
    _compute_edge_weights(centers, port::DeltaGapArrayPort)

Compute edge weights by ESTIMATING dimensions from edge positions (for arbitrary shapes).

This method estimates width/height from the bounding box of edge centers, which may
be inaccurate. Use `_compute_edge_weights(centers, port, width, height)` when dimensions
are known (e.g., for rectangular ports).
"""
function _compute_edge_weights(centers::Vector{MVec3D{FT}}, port::DeltaGapArrayPort) where {FT}
    # Estimate dimensions from actual edge positions
    width, height = _estimate_dimensions_from_edges(centers, port.center, port.widthDir, port.heightDir)
    
    # Fall back to explicit dimension method
    return _compute_edge_weights(centers, port, width, height)
end

function _estimate_dimensions_from_edges(centers, port_center, wdir, hdir)
    FT = eltype(centers[1])
    max_u, max_v = zero(FT), zero(FT)
    for c in centers
        rel = c - port_center
        max_u = max(max_u, abs(rel ⋅ wdir))
        max_v = max(max_v, abs(rel ⋅ hdir))
    end
    return 2 * max_u, 2 * max_v
end

# _excitation_array! and _excitation_single_edge! are implemented in MoM_Kernels.jl (SurfacePortExcitation.jl)


# =============================================================================
# Convenience Shape Constructors
# =============================================================================

"""
    RectangularDeltaGapPort(; center, normal, width, height, forcePortPEC=true, kwargs...)

Create a rectangular port, bound to mesh immediately.

This is a convenience constructor for the common rectangular case.
For deferred binding, use `DeltaGapArrayPort` + `bind_to_mesh!`.

# Keyword Arguments
- `forcePortPEC::Bool = true` -- Set surface impedance `Zs = 0` (PEC) on port triangles.
  See [`bind_to_mesh!`](@ref) for details.
"""
function RectangularDeltaGapPort(;
    center::AbstractVector{FT},
    normal::AbstractVector{FT},
    width::FT,
    height::FT,
    widthDir::AbstractVector{FT} = zeros(FT, 3),
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}},
    forcePortPEC::Bool = true,
    kwargs...
) where {FT<:Real, IT<:Integer}
    
    center_m = MVec3D{FT}(center)
    normal_m = MVec3D{FT}(normal)
    widthDir_m = MVec3D{FT}(widthDir)
    
    # Create unbound port
    port = DeltaGapArrayPort{FT, IT}(;
        center = center_m,
        normal = normal_m,
        widthDir = widthDir_m,
        kwargs...
    )
    
    # Create rectangular predicate
    n̂, wdir, hdir = _compute_port_frame(normal_m, widthDir_m)
    predicate = let c=center_m, w=wdir, h=hdir, hw=width/2, hh=height/2
        p -> abs((p - c) ⋅ w) <= hw && abs((p - c) ⋅ h) <= hh
    end
    
    # Bind to mesh
    bind_to_mesh!(port, predicate, trianglesInfo, rwgsInfo; 
                  estimateDimensions=false, forcePortPEC=forcePortPEC)
    
    # Set known dimensions
    port.modeImpedance = _compute_mode_impedance_generic(
        port.excitationDistribution, port.freq, width, height
    )
    
    return port
end

"""
    CircularDeltaGapPort(; center, normal, radius, forcePortPEC=true, kwargs...)

Create a circular port, bound to mesh immediately.

# Keyword Arguments
- `forcePortPEC::Bool = true` -- Set surface impedance `Zs = 0` (PEC) on port triangles.
  See [`bind_to_mesh!`](@ref) for details.
"""
function CircularDeltaGapPort(;
    center::AbstractVector{FT},
    normal::AbstractVector{FT},
    radius::FT,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}},
    forcePortPEC::Bool = true,
    kwargs...
) where {FT<:Real, IT<:Integer}
    
    center_m = MVec3D{FT}(center)
    normal_m = MVec3D{FT}(normal)
    
    port = DeltaGapArrayPort{FT, IT}(;
        center = center_m,
        normal = normal_m,
        kwargs...
    )
    
    n̂ = _normalize_port_vector(normal_m)
    predicate = let c=center_m, n=n̂, r=radius
        p -> begin
            rel = MVec3D{FT}(p) - c
            normal_dist = abs(rel ⋅ n)
            in_plane = normal_dist <= r * FT(1e-4)
            in_plane || return false
            radial_dist = norm(rel - normal_dist * n)
            return radial_dist <= r
        end
    end
    
    bind_to_mesh!(port, predicate, trianglesInfo, rwgsInfo; forcePortPEC=forcePortPEC)
    return port
end


# =============================================================================
# Utility Methods
# =============================================================================

"""
    get_edge_info(port::DeltaGapArrayPort, rwgID::Integer)

Get information about a specific boundary edge.

Returns a NamedTuple or `nothing` if not found.
"""
function get_edge_info(port::DeltaGapArrayPort{FT, IT, DT}, rwgID::Integer) where {FT, IT, DT}
    port.isBound || return nothing
    idx = findfirst(==(IT(rwgID)), port.rwgIDs)
    idx === nothing && return nothing
    
    return (
        rwgID = port.rwgIDs[idx],
        weight = port.edgeWeights[idx],
        length = port.edgeLengths[idx],
        center = port.edgeCenters[idx],
        orient = port.edgeOrient[idx],
        tri_pos = port.triID_pos[idx],
        tri_neg = port.triID_neg[idx]
    )
end

"""
    get_port_power(port::DeltaGapArrayPort, Z_matrix, V_excitation)

Compute injected power: P = 0.5 * Re{V * I*}.
"""
function get_port_power(port::DeltaGapArrayPort{FT, IT, DT},
                        Z_matrix::AbstractMatrix{Complex{FT}},
                        V_excitation::AbstractVector{Complex{FT}}) where {FT, IT, DT}
    port.isBound || error("Port must be bound to mesh")
    Z_in = computeInputImpedance(port, Z_matrix, V_excitation)
    I_port = port.V / Z_in
    return FT(0.5) * real(port.V * conj(I_port))
end

"""
    get_port_perimeter(port::DeltaGapArrayPort)

Total perimeter length (sum of all boundary edges).
"""
function get_port_perimeter(port::DeltaGapArrayPort{FT, IT, DT}) where {FT, IT, DT}
    port.isBound || return zero(FT)
    return sum(port.edgeLengths)
end

"""
    get_port_area(port::DeltaGapArrayPort)

Cross-sectional area (sum of triangle areas in port region).
"""
function get_port_area(port::DeltaGapArrayPort{FT, IT, DT}, trianglesInfo) where {FT, IT, DT}
    port.isBound || return zero(FT)
    tri_set = Set(port.triangleIDs)
    area = zero(FT)
    for tri in trianglesInfo
        if tri.triID in tri_set
            area += tri.area
        end
    end
    return area
end
