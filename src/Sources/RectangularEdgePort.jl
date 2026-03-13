
"""
    RectangularEdgePort{FT<:Real, IT<:Integer, DT<:AbstractExcitationDistribution{FT}} <: PortType

Rectangular delta-gap port excitation for PEC surfaces with configurable mode distribution.

This implementation uses **composition** with `DeltaGapArrayPort` as its base,
delegating all mesh binding and excitation computation to the generic implementation.
Field access is transparently delegated to the base port via `getproperty`/`setproperty!`.

# Fields (Own)
- `base::DeltaGapArrayPort{FT, IT, DT}` -- The underlying delta-gap array port
- `width::FT` -- Port width
- `height::FT` -- Port height  
- `tol::FT` -- Geometric tolerance for vertex selection

# Delegated Fields (from `base`)
`id`, `V`, `freq`, `portType`, `excitationDistribution`, `modeImpedance`,
`center`, `normal`, `widthDir`, `heightDir`, `isBound`, `vertexIDs`,
`triangleIDs`, `rwgIDs`, `triID_pos`, `triID_neg`, `edgeLengths`,
`edgeCenters`, `edgeOrient`, `edgeWeights`, `singleEdgeMode`, `primaryRwgID`, `isActive`
"""
mutable struct RectangularEdgePort{
    FT<:Real,
    IT<:Integer,
    DT<:AbstractExcitationDistribution{FT}
} <: PortType
    base::DeltaGapArrayPort{FT, IT, DT}
    width::FT
    height::FT
    tol::FT
    mode::Symbol  # Deprecated field for backward compatibility
end


# =============================================================================
# Property Delegation to Base DeltaGapArrayPort
# =============================================================================

const _RECTANGULAR_EDGE_PORT_OWN_FIELDS = (:base, :width, :height, :tol, :mode)

@inline function Base.getproperty(port::RectangularEdgePort, name::Symbol)
    if name in _RECTANGULAR_EDGE_PORT_OWN_FIELDS
        return getfield(port, name)
    else
        return getproperty(getfield(port, :base), name)
    end
end

@inline function Base.setproperty!(port::RectangularEdgePort, name::Symbol, value)
    if name in _RECTANGULAR_EDGE_PORT_OWN_FIELDS
        setfield!(port, name, value)
    else
        setproperty!(getfield(port, :base), name, value)
    end
    return port
end

@inline function Base.propertynames(port::RectangularEdgePort, private::Bool=false)
    own = _RECTANGULAR_EDGE_PORT_OWN_FIELDS
    base = propertynames(getfield(port, :base), private)
    return (own..., base...)
end


# =============================================================================
# Constructor
# =============================================================================

"""
    RectangularEdgePort{FT, IT}(;
        id, V, freq, center, normal, width, height, widthDirection, tol,
        excitationDistribution, mode, trianglesInfo, rwgsInfo, isActive
    )

Construct a rectangular edge port with configurable excitation distribution.

The port is created with a `DeltaGapArrayPort` as its base, delegating mesh binding
and excitation computation. Rectangular-specific parameters (`width`, `height`, `tol`)
are stored directly on the `RectangularEdgePort`.

# Arguments
- `id::IT` -- Port identifier (default: 0)
- `V::Complex{FT}` -- Excitation voltage (default: 1.0)
- `freq::FT` -- Operating frequency in Hz (default: 0)
- `center::MVec3D{FT}` -- Port center position (required)
- `normal::MVec3D{FT}` -- Port normal vector (required)
- `width::FT` -- Port width (required)
- `height::FT` -- Port height (required)
- `widthDirection::MVec3D{FT}` -- Direction along width (default: auto-computed)
- `tol::FT` -- Geometric tolerance (default: 1e-6)
- `excitationDistribution` -- Voltage distribution pattern (default: `UniformDistribution()`)
- `mode` -- **Deprecated**: Legacy mode parameter. Use `excitationDistribution` instead.
- `trianglesInfo` -- Triangle mesh information (required)
- `rwgsInfo` -- RWG basis function information (required)
- `isActive::Bool` -- Whether port is active (default: true)
"""
function RectangularEdgePort{FT, IT}(;
    id::IT = zero(IT),
    V::Complex{FT} = one(Complex{FT}),
    freq::FT = zero(FT),
    center::MVec3D{FT},
    normal::MVec3D{FT},
    width::FT,
    height::FT,
    widthDirection::MVec3D{FT} = zero(MVec3D{FT}),
    tol::FT = FT(1e-6),
    excitationDistribution::AbstractExcitationDistribution{FT} = UniformDistribution(),
    mode::Union{Symbol, Nothing} = nothing,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}},
    isActive::Bool = true
) where {FT<:Real, IT<:Integer}

    # Handle legacy mode parameter
    actual_distribution = excitationDistribution
    if mode !== nothing
        @warn "The `mode` keyword argument is deprecated. Use `excitationDistribution=UniformDistribution()` instead."
        mode == :TE10 || error("Only :TE10 mode is supported in legacy mode. Use `excitationDistribution` for other distributions.")
        actual_distribution = UniformDistribution{FT}()
    end

    # Step 1: Compute port coordinate frame
    n̂, wdir, hdir = _compute_port_frame(normal, widthDirection)
    
    # Step 2: Create rectangular predicate for edge discovery
    predicate = let c = center, w = wdir, h = hdir, hw = width / 2, hh = height / 2
        p -> abs((p - c) ⋅ w) <= hw && abs((p - c) ⋅ h) <= hh
    end

    # Step 3: Find boundary edges (geometry discovery only)
    # Using the low-level function that doesn't compute weights
    boundary_data = find_port_boundary_edges(
        predicate, center, normal, trianglesInfo, rwgsInfo
    )
    
    # Step 4: Collect vertex and triangle IDs for the port
    vertex_tol = _compute_tolerance(trianglesInfo)
    vertex_ids = _collect_vertices_by_predicate(predicate, center, normal, 
                                                  vertex_tol, trianglesInfo)
    triangle_ids = _collect_triangles_by_vertices(vertex_ids, trianglesInfo)

    # Step 5: Create unbound DeltaGapArrayPort (base port)
    gap_port = DeltaGapArrayPort{FT, IT}(
        id,
        V,
        freq,
        :rectangular_edge,  # Port type for rectangular edge ports
        actual_distribution,
        Complex{FT}(Inf),    # modeImpedance (computed below)
        center,
        normal,
        wdir,
        hdir,
        true,                # isBound - we're setting everything now
        vertex_ids,
        triangle_ids,
        boundary_data.rwgIDs,
        boundary_data.triPos,
        boundary_data.triNeg,
        boundary_data.lengths,
        boundary_data.centers,
        boundary_data.orients,
        Complex{FT}[],       # edgeWeights - computed next with KNOWN dimensions
        false,               # singleEdgeMode
        zero(IT),            # primaryRwgID
        isActive
    )
    
    # Step 6: Compute edge weights using KNOWN dimensions (not estimated!)
    # This is the key fix: we use width and height from constructor parameters
    gap_port.edgeWeights = _compute_edge_weights(
        boundary_data.centers, gap_port, width, height
    )
    
    # Step 7: Compute mode impedance with known dimensions
    gap_port.modeImpedance = _compute_mode_impedance_generic(
        actual_distribution, freq, width, height
    )

    # Step 8: Determine mode symbol for backward compatibility
    mode_symbol = mode === nothing ? :custom : mode

    # Step 9: Create RectangularEdgePort wrapping the fully-initialized base port
    return RectangularEdgePort{FT, IT, typeof(actual_distribution)}(
        gap_port,
        width,
        height,
        tol,
        mode_symbol
    )
end

RectangularEdgePort(args...; kwargs...) = RectangularEdgePort{Precision.FT, IntDtype}(args...; kwargs...)


# =============================================================================
# Internal: Extract Base DeltaGapArrayPort
# =============================================================================

"""
    _to_delta_gap_array_port(port::RectangularEdgePort)

Extract the underlying `DeltaGapArrayPort` from a `RectangularEdgePort`.

With the composition-based design, this simply returns the `base` field.
This function is used by MoM_Kernels for delegation to DeltaGapArrayPort methods.
"""
@inline function _to_delta_gap_array_port(port::RectangularEdgePort{FT, IT, DT}) where {FT, IT, DT}
    return getfield(port, :base)
end


# =============================================================================
# Configuration Methods (Specialized for RectangularEdgePort)
# =============================================================================

"""
    set_excitation_distribution!(port::RectangularEdgePort, distribution)

Change the excitation distribution for a rectangular edge port.

Unlike the generic `DeltaGapArrayPort` version, this uses the STORED `width` and `height`
from the port (which are known exactly at construction), avoiding estimation errors.
"""
function set_excitation_distribution!(
    port::RectangularEdgePort{FT, IT, DT},
    distribution::AbstractExcitationDistribution{FT}
) where {FT, IT, DT}
    # Update distribution on base port
    port.base.excitationDistribution = distribution
    
    # Recompute edge weights using STORED dimensions (not estimated!)
    if port.base.isBound
        port.base.edgeWeights = _compute_edge_weights(
            port.base.edgeCenters, port.base, port.width, port.height
        )
    end
    
    # Update mode impedance using STORED dimensions
    port.base.modeImpedance = _compute_mode_impedance_generic(
        distribution, port.base.freq, port.width, port.height
    )
    
    return port
end


# =============================================================================
# Source Field Methods: Delegated to DeltaGapArrayPort equivalents
# =============================================================================

function sourceEfield(port::RectangularEdgePort{FT, IT, DT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer, DT}
    return sourceEfield(port.base, r)
end

function sourceHfield(port::RectangularEdgePort{FT, IT, DT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer, DT}
    return sourceHfield(port.base, r)
end


# =============================================================================
# Port Voltage and Current Methods
# =============================================================================

function getPortVoltage(port::RectangularEdgePort{FT, IT, DT}, current::Complex{FT}) where {FT<:Real, IT<:Integer, DT}
    return getPortVoltage(port.base, current)
end

function getPortCurrent(port::RectangularEdgePort{FT, IT, DT}; Z::Complex{FT} = Complex{FT}(50.0)) where {FT<:Real, IT<:Integer, DT}
    return getPortCurrent(port.base; Z = Z)
end


# =============================================================================
# S-Parameter Methods
# =============================================================================
# NOTE: computeInputImpedance and computeS11 for RectangularEdgePort 
# are now defined in MoM_Kernels.jl for all port types to be consistent 
# with the architecture. This avoids duplication and keeps computational 
# algorithms in MoM_Kernels.


