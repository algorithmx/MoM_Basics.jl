
"""
    RectangularWaveguidePort{FT<:Real, IT<:Integer, DT<:AbstractExcitationDistribution{FT}} <: PortType

Rectangular delta-gap port excitation for PEC surfaces with configurable mode distribution.

The user provides a rectangle plus tolerance that matches the mesh. The rectangle is
expanded into a thin box to filter mesh vertices, selected triangles are inferred from
those vertices, and the port edge RWGs are identified from the selected patch boundary.

# Fields
- `id` -- Port identifier
- `V` -- Excitation voltage
- `freq` -- Operating frequency in Hz
- `portType` -- Port type symbol (`:rectangular_waveguide`)
- `excitationDistribution` -- Mode distribution (e.g., `TE10()`, `ModalDistribution(:TE, 2, 1)`)
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

"""
    RectangularWaveguidePort{FT, IT}(;
        id, V, freq, center, normal, width, height, widthDirection, tol,
        excitationDistribution, trianglesInfo, rwgsInfo, isActive
    )

Construct a rectangular waveguide port with configurable excitation distribution.

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
- `excitationDistribution` -- Mode distribution (default: `TE10()`)
- `trianglesInfo` -- Triangle mesh information (required)
- `rwgsInfo` -- RWG basis function information (required)
- `isActive::Bool` -- Whether port is active (default: true)

# Example
```julia
# TE₁₀ mode (default)
port = RectangularWaveguidePort(;
    center = [0, 0, 0], normal = [0, 0, 1],
    width = 0.02286, height = 0.01016,  # WR-90
    freq = 10e9,
    trianglesInfo = trianglesInfo, rwgsInfo = rwgsInfo
)

# TE₂₀ mode
port = RectangularWaveguidePort(;
    excitationDistribution = ModalDistribution(:TE, 2, 0),
    center = [0, 0, 0], normal = [0, 0, 1],
    width = 0.02286, height = 0.01016,
    freq = 15e9,
    trianglesInfo = trianglesInfo, rwgsInfo = rwgsInfo
)
```
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
    excitationDistribution::AbstractExcitationDistribution{FT} = TE10(; FT=FT),
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}},
    isActive::Bool = true
) where {FT<:Real, IT<:Integer}
    n̂, wdir, hdir = _rectangular_port_axes(normal, widthDirection)
    vertex_ids = _collect_rectangular_port_vertices(center, wdir, hdir, n̂, width, height, tol, trianglesInfo)
    isempty(vertex_ids) && error("No mesh vertices found inside the rectangular port box")

    triangle_ids = _collect_rectangular_port_triangles(vertex_ids, trianglesInfo)
    isempty(triangle_ids) && error("No triangles found for the selected rectangular port region")

    rwg_ids, tri_id_pos, tri_id_neg, edge_lengths, edge_centers, edge_orients, edge_weights =
        _collect_rectangular_port_edges(vertex_ids, triangle_ids, center, wdir, hdir, n̂, width, height,
                                        excitationDistribution, rwgsInfo, trianglesInfo)
    isempty(rwg_ids) && error("No RWG edges found on the rectangular port boundary")

    # Compute mode impedance
    mode_impedance = if freq > zero(FT) && excitationDistribution isa ModalDistribution{FT}
        Complex{FT}(compute_mode_impedance(excitationDistribution, freq, width, height))
    else
        Complex{FT}(Inf)  # Unknown impedance
    end

    # Extract mode symbol for backward compatibility
    mode_symbol = if excitationDistribution isa ModalDistribution{FT}
        Symbol(excitationDistribution.modeType, excitationDistribution.m, excitationDistribution.n)
    else
        :custom
    end

    return RectangularWaveguidePort{FT, IT, typeof(excitationDistribution)}(
        id,
        Complex{FT}(V),
        freq,
        :rectangular_waveguide,
        excitationDistribution,
        mode_impedance,
        center,
        n̂,
        wdir,
        hdir,
        width,
        height,
        tol,
        isActive,
        vertex_ids,
        triangle_ids,
        rwg_ids,
        tri_id_pos,
        tri_id_neg,
        edge_lengths,
        edge_centers,
        edge_orients,
        edge_weights,
        mode_symbol
    )
end

"""
    RectangularWaveguidePort{FT, IT}(; mode::Symbol=:TE10, kwargs...)

Legacy constructor with `mode::Symbol` for backward compatibility.

**Deprecated**: Use `excitationDistribution` parameter instead.
Supported modes: `:TE10` (only `:TE10` is supported in legacy mode).
"""
function RectangularWaveguidePort{FT, IT}(;
    mode::Symbol = :TE10,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}},
    kwargs...
) where {FT<:Real, IT<:Integer}
    # Issue deprecation warning
    @warn "The `mode` keyword argument is deprecated. Use `excitationDistribution=TE10()` instead."

    mode == :TE10 || error("Only :TE10 mode is supported in legacy mode. Use `excitationDistribution` for other modes.")

    # Convert mode symbol to distribution
    distribution = TE10(; FT=FT)

    RectangularWaveguidePort{FT, IT}(;
        excitationDistribution = distribution,
        trianglesInfo = trianglesInfo,
        rwgsInfo = rwgsInfo,
        kwargs...
    )
end

RectangularWaveguidePort(args...; kwargs...) = RectangularWaveguidePort{Precision.FT, IntDtype}(args...; kwargs...)

# ---------------------------------------------------



function sourceEfield(port::RectangularWaveguidePort{FT, IT, DT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer, DT}
    return zero(MVec3D{Complex{FT}})
end

function sourceHfield(port::RectangularWaveguidePort{FT, IT, DT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer, DT}
    return zero(MVec3D{Complex{FT}})
end

function excitationVectorEFIE(
    port::RectangularWaveguidePort{FT, IT, DT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer
) where {FT<:Real, IT<:Integer, DT}
    V = zeros(Complex{FT}, nbf)
    excitationVectorEFIE!(V, port, trianglesInfo)
    return V
end

function excitationVectorEFIE!(
    V::Vector{Complex{FT}},
    port::RectangularWaveguidePort{FT, IT, DT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer, DT}
    for edgei in eachindex(port.rwgIDs)
        rwgID = port.rwgIDs[edgei]
        rwgID <= length(V) || continue
        edge_voltage = port.V * port.edgeWeights[edgei]
        if port.triID_neg[edgei] > 0
            V[rwgID] += edge_voltage * port.edgeLengths[edgei] / 2
        else
            V[rwgID] += edge_voltage * port.edgeLengths[edgei]
        end
    end
    return V
end

function excitationVectorCFIE(
    port::RectangularWaveguidePort{FT, IT, DT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    alpha::FT = FT(0.5),
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, DT}
    return excitationVectorEFIE(port, trianglesInfo, nbf)
end

function getPortVoltage(port::RectangularWaveguidePort{FT, IT, DT}, current::Complex{FT}) where {FT<:Real, IT<:Integer, DT}
    return port.V
end

function getPortCurrent(port::RectangularWaveguidePort{FT, IT, DT}; Z::Complex{FT} = 50.0) where {FT<:Real, IT<:Integer, DT}
    return port.V / Z
end

"""
    computeInputImpedance(port, Z_matrix, V_excitation) -> Complex{FT}

Compute the input impedance at the waveguide port.

Uses the MoM solution to compute the input impedance from the excitation voltage
and the induced currents at the port edges.

# Arguments
- `port` -- RectangularWaveguidePort instance
- `Z_matrix` -- MoM impedance matrix
- `V_excitation` -- Excitation vector

# Returns
- Complex input impedance in Ohms

# Algorithm
The input impedance is computed as:
```
Z_in = V / I_port
```
where the port current is obtained by summing the weighted RWG currents:
```
I_port = Σ I_n × w_n × l_n/2 (or l_n for single-sided RWGs)
```
"""
function computeInputImpedance(
    port::RectangularWaveguidePort{FT, IT, DT},
    Z_matrix::AbstractMatrix{Complex{FT}},
    V_excitation::AbstractVector{Complex{FT}}
) where {FT<:Real, IT<:Integer, DT}
    # Solve for current coefficients
    I_coeff = Z_matrix \ V_excitation

    # Compute port current by summing weighted contributions
    I_port = zero(Complex{FT})
    for i in eachindex(port.rwgIDs)
        rwgID = port.rwgIDs[i]
        rwgID <= length(I_coeff) || continue

        # Get current coefficient for this RWG
        I_n = I_coeff[rwgID]

        # Weight by edge weight and length
        # For RWGs with positive triangle inside port: use l/2
        # For RWGs with negative triangle inside port: use l
        weight_factor = port.triID_neg[i] > 0 ? port.edgeLengths[i] / 2 : port.edgeLengths[i]

        I_port += I_n * port.edgeWeights[i] * weight_factor
    end

    # Input impedance
    Z_in = port.V / I_port

    return Z_in
end

"""
    computeS11(port, Z_matrix, V_excitation; Z0=nothing) -> Complex{FT}

Compute the S₁₁ reflection coefficient at the waveguide port.

# Arguments
- `port` -- RectangularWaveguidePort instance
- `Z_matrix` -- MoM impedance matrix
- `V_excitation` -- Excitation vector
- `Z0` -- Optional reference impedance (default: use mode impedance)

# Returns
- Complex S₁₁ reflection coefficient

# Algorithm
The reflection coefficient is computed as:
```
S₁₁ = (Z_in - Z_ref) / (Z_in + Z_ref)
```
where `Z_ref` is the reference impedance (mode impedance by default).
"""
function computeS11(
    port::RectangularWaveguidePort{FT, IT, DT},
    Z_matrix::AbstractMatrix{Complex{FT}},
    V_excitation::AbstractVector{Complex{FT}};
    Z0::Union{FT, Complex{FT}, Nothing} = nothing
) where {FT<:Real, IT<:Integer, DT}
    # Determine reference impedance
    Z_ref = if Z0 !== nothing
        Complex{FT}(Z0)
    elseif isfinite(port.modeImpedance)
        port.modeImpedance
    else
        Complex{FT}(50.0)  # Fallback to 50 Ohm
    end

    Z_in = computeInputImpedance(port, Z_matrix, V_excitation)

    return (Z_in - Z_ref) / (Z_in + Z_ref)
end
