
"""
    RectangularWaveguidePort{FT<:Real, IT<:Integer, DT<:AbstractExcitationDistribution{FT}} <: PortType

Rectangular delta-gap port excitation for PEC surfaces with configurable mode distribution.

This implementation uses `DeltaGapArrayPort` as its base, delegating all mesh binding
and excitation computation to the generic implementation while maintaining full
backward compatibility with the original interface.

# Fields
- `id` -- Port identifier
- `V` -- Excitation voltage
- `freq` -- Operating frequency in Hz
- `portType` -- Port type symbol (`:rectangular_waveguide`)
- `excitationDistribution` -- Voltage distribution pattern (e.g., `UniformDistribution()`, `SingleSideDistribution(:left)`)
- `modeImpedance` -- Characteristic impedance of the mode at `freq`
- `center` -- Port center position
- `normal` -- Port normal vector
- `widthDir` -- Unit vector along port width
- `heightDir` -- Unit vector along port height
- `width` -- Port width
- `height` -- Port height
- `tol` -- Geometric tolerance for vertex selection
- `isActive` -- Whether port is active in simulation
- `vertexIDs` -- Mesh vertex IDs in port region
- `triangleIDs` -- Triangle IDs in port region
- `rwgIDs` -- RWG basis function IDs on port boundary
- `triID_pos` -- Positive triangle IDs for RWGs
- `triID_neg` -- Negative triangle IDs for RWGs
- `edgeLengths` -- Edge lengths of port boundary edges
- `edgeCenters` -- Edge center positions
- `edgeOrient` -- Edge orientation vectors
- `edgeWeights` -- Complex voltage weights for each edge
- `mode` -- Deprecated field for backward compatibility
"""
mutable struct RectangularWaveguidePort{
    FT<:Real,
    IT<:Integer,
    DT<:AbstractExcitationDistribution{FT}
} <: PortType
    id                      ::IT
    V                       ::Complex{FT}
    freq                    ::FT
    portType                ::Symbol
    excitationDistribution  ::DT
    modeImpedance           ::Complex{FT}
    center                  ::MVec3D{FT}
    normal                  ::MVec3D{FT}
    widthDir                ::MVec3D{FT}
    heightDir               ::MVec3D{FT}
    width                   ::FT
    height                  ::FT
    tol                     ::FT
    isActive                ::Bool
    vertexIDs               ::Vector{IT}
    triangleIDs             ::Vector{IT}
    rwgIDs                  ::Vector{IT}
    triID_pos               ::Vector{IT}
    triID_neg               ::Vector{IT}
    edgeLengths             ::Vector{FT}
    edgeCenters             ::Vector{MVec3D{FT}}
    edgeOrient              ::Vector{MVec3D{FT}}
    edgeWeights             ::Vector{Complex{FT}}
    # Deprecated fields for backward compatibility
    mode                    ::Symbol
end


# =============================================================================
# Internal: Convert RectangularWaveguidePort <-> DeltaGapArrayPort
# =============================================================================

"""
    _to_delta_gap_array_port(port::RectangularWaveguidePort)

Convert RectangularWaveguidePort to DeltaGapArrayPort for delegation.
This is a view conversion - no data is copied, just wrapped.
"""
function _to_delta_gap_array_port(port::RectangularWaveguidePort{FT, IT, DT}) where {FT, IT, DT}
    # Create a DeltaGapArrayPort that shares all data with the original
    # This avoids any copying overhead
    return DeltaGapArrayPort{FT, IT, DT}(
        port.id,
        port.V,
        port.freq,
        :rectangular_waveguide,  # Different portType but that's fine
        port.excitationDistribution,
        port.modeImpedance,
        port.center,
        port.normal,
        port.widthDir,
        port.heightDir,
        true,  # isBound - rectangular port is always bound
        port.vertexIDs,
        port.triangleIDs,
        port.rwgIDs,
        port.triID_pos,
        port.triID_neg,
        port.edgeLengths,
        port.edgeCenters,
        port.edgeOrient,
        port.edgeWeights,
        false,  # singleEdgeMode
        zero(IT),  # primaryRwgID
        port.isActive
    )
end

# =============================================================================
# Constructor: Delegates to DeltaGapArrayPort
# =============================================================================

"""
    RectangularWaveguidePort{FT, IT}(;
        id, V, freq, center, normal, width, height, widthDirection, tol,
        excitationDistribution, mode, trianglesInfo, rwgsInfo, isActive
    )

Construct a rectangular waveguide port with configurable excitation distribution.

**Implementation Note**: This constructor internally uses `DeltaGapArrayPort`
to perform mesh binding and edge identification, ensuring functional equivalence
with the generic implementation.

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
function RectangularWaveguidePort{FT, IT}(;
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

    # Step 1: Create a DeltaGapArrayPort with rectangular binding
    # This does all the heavy lifting: vertex collection, triangle identification,
    # boundary edge finding, and weight computation
    gap_port = DeltaGapArrayPort{FT, IT}(;
        id = id,
        V = V,
        freq = freq,
        center = center,
        normal = normal,
        widthDir = widthDirection,
        excitationDistribution = actual_distribution,
        isActive = isActive
    )

    # Step 2: Create rectangular predicate
    n̂, wdir, hdir = _compute_port_frame(normal, widthDirection)
    predicate = let c = center, w = wdir, h = hdir, hw = width / 2, hh = height / 2
        p -> abs((p - c) ⋅ w) <= hw && abs((p - c) ⋅ h) <= hh
    end

    # Step 3: Bind to mesh using DeltaGapArrayPort's generic binding
    bind_to_mesh!(gap_port, predicate, trianglesInfo, rwgsInfo; estimateDimensions = false)

    # Step 4: Mode impedance computation (Inf for non-modal distributions)
    mode_impedance = Complex{FT}(Inf)

    # Step 5: Mode symbol for backward compatibility
    mode_symbol = mode === nothing ? :custom : mode

    # Step 6: Create RectangularWaveguidePort with data from DeltaGapArrayPort
    return RectangularWaveguidePort{FT, IT, typeof(actual_distribution)}(
        gap_port.id,
        gap_port.V,
        gap_port.freq,
        :rectangular_waveguide,
        gap_port.excitationDistribution,
        mode_impedance,
        gap_port.center,
        gap_port.normal,
        gap_port.widthDir,
        gap_port.heightDir,
        width,
        height,
        tol,
        gap_port.isActive,
        gap_port.vertexIDs,
        gap_port.triangleIDs,
        gap_port.rwgIDs,
        gap_port.triID_pos,
        gap_port.triID_neg,
        gap_port.edgeLengths,
        gap_port.edgeCenters,
        gap_port.edgeOrient,
        gap_port.edgeWeights,
        mode_symbol
    )
end

RectangularWaveguidePort(args...; kwargs...) = RectangularWaveguidePort{Precision.FT, IntDtype}(args...; kwargs...)


# =============================================================================
# Source Field Methods: Delegated to DeltaGapArrayPort equivalents
# =============================================================================

function sourceEfield(port::RectangularWaveguidePort{FT, IT, DT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer, DT}
    return zero(MVec3D{Complex{FT}})
end

function sourceHfield(port::RectangularWaveguidePort{FT, IT, DT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer, DT}
    return zero(MVec3D{Complex{FT}})
end


# Note: excitationVectorEFIE, excitationVectorMFIE, excitationVectorCFIE
# are now implemented in MoM_Kernels.jl/src/ZmatAndVvec/Ports/SurfacePortExcitation.jl
# for consistency with the consumer-provider architecture where MoM_Kernels
# provides all matrix/vector computation.


# =============================================================================
# Port Voltage and Current Methods
# =============================================================================

function getPortVoltage(port::RectangularWaveguidePort{FT, IT, DT}, current::Complex{FT}) where {FT<:Real, IT<:Integer, DT}
    return port.V
end

function getPortCurrent(port::RectangularWaveguidePort{FT, IT, DT}; Z::Complex{FT} = Complex{FT}(50.0)) where {FT<:Real, IT<:Integer, DT}
    return port.V / Z
end


# =============================================================================
# S-Parameter Methods
# =============================================================================
# NOTE: computeInputImpedance and computeS11 for RectangularWaveguidePort 
# are now defined in MoM_Kernels.jl for all port types to be consistent 
# with the architecture. This avoids duplication and keeps computational 
# algorithms in MoM_Kernels.
