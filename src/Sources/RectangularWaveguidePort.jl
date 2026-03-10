
"""
    RectangularWaveguidePort{FT<:Real, IT<:Integer} <: ExcitingSource

Rectangular delta-gap port excitation for PEC surfaces.

The user provides a rectangle plus tolerance that matches the mesh. The rectangle is
expanded into a thin box to filter mesh vertices, selected triangles are inferred from
those vertices, and the port edge RWGs are identified from the selected patch boundary.
"""
mutable struct RectangularWaveguidePort{FT<:Real, IT<:Integer} <: ExcitingSource
    id              ::IT
    V               ::Complex{FT}
    freq            ::FT
    portType        ::Symbol
    mode            ::Symbol
    center          ::MVec3D{FT}
    normal          ::MVec3D{FT}
    widthDir        ::MVec3D{FT}
    heightDir       ::MVec3D{FT}
    width           ::FT
    height          ::FT
    tol             ::FT
    isActive        ::Bool
    vertexIDs       ::Vector{IT}
    triangleIDs     ::Vector{IT}
    rwgIDs          ::Vector{IT}
    triID_pos       ::Vector{IT}
    triID_neg       ::Vector{IT}
    edgeLengths     ::Vector{FT}
    edgeCenters     ::Vector{MVec3D{FT}}
    edgeOrient      ::Vector{MVec3D{FT}}
    edgeWeights     ::Vector{Complex{FT}}
end

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
    mode::Symbol = :TE10,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}},
    isActive::Bool = true
) where {FT<:Real, IT<:Integer}
    mode == :TE10 || error("Only :TE10 mode is implemented for RectangularWaveguidePort")

    n̂, wdir, hdir = _rectangular_port_axes(normal, widthDirection)
    vertex_ids = _collect_rectangular_port_vertices(center, wdir, hdir, n̂, width, height, tol, trianglesInfo)
    isempty(vertex_ids) && error("No mesh vertices found inside the rectangular port box")

    triangle_ids = _collect_rectangular_port_triangles(vertex_ids, trianglesInfo)
    isempty(triangle_ids) && error("No triangles found for the selected rectangular port region")

    rwg_ids, tri_id_pos, tri_id_neg, edge_lengths, edge_centers, edge_orients, edge_weights =
        _collect_rectangular_port_edges(vertex_ids, triangle_ids, center, wdir, hdir, n̂, width, rwgsInfo, trianglesInfo)
    isempty(rwg_ids) && error("No RWG edges found on the rectangular port boundary")

    return RectangularWaveguidePort{FT, IT}(
        id,
        Complex{FT}(V),
        freq,
        :rectangular_waveguide,
        mode,
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
        edge_weights
    )
end

RectangularWaveguidePort(args...; kwargs...) = RectangularWaveguidePort{Precision.FT, IntDtype}(args...; kwargs...)

# ---------------------------------------------------



function sourceEfield(port::RectangularWaveguidePort{FT, IT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer}
    return zero(MVec3D{Complex{FT}})
end

function sourceHfield(port::RectangularWaveguidePort{FT, IT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer}
    return zero(MVec3D{Complex{FT}})
end

function excitationVectorEFIE(
    port::RectangularWaveguidePort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer
) where {FT<:Real, IT<:Integer}
    V = zeros(Complex{FT}, nbf)
    excitationVectorEFIE!(V, port, trianglesInfo)
    return V
end

function excitationVectorEFIE!(
    V::Vector{Complex{FT}},
    port::RectangularWaveguidePort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
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
    port::RectangularWaveguidePort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    alpha::FT = FT(0.5),
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer}
    return excitationVectorEFIE(port, trianglesInfo, nbf)
end

function getPortVoltage(port::RectangularWaveguidePort{FT, IT}, current::Complex{FT}) where {FT<:Real, IT<:Integer}
    return port.V
end

function getPortCurrent(port::RectangularWaveguidePort{FT, IT}; Z::Complex{FT} = 50.0) where {FT<:Real, IT<:Integer}
    return port.V / Z
end

function computeInputImpedance(
    port::RectangularWaveguidePort{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer}
    error("Modal impedance extraction for RectangularWaveguidePort is not implemented yet")
end

function computeS11(
    port::RectangularWaveguidePort{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer}
    error("Modal S-parameter extraction for RectangularWaveguidePort is not implemented yet")
end

function getPortImpedance(
    port::RectangularWaveguidePort{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}}
) where {FT<:Real, IT<:Integer}
    return computeInputImpedance(port, Z_matrix, V_excitation)
end
