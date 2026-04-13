
"""
    PortArray{FT, IT, PT}

Container for one or more ports used in a MoM simulation.

The type parameter `PT` is the *element type* of the internal `ports` vector.  For
homogeneous collections (all ports share the same concrete type) `PT` is that concrete
type, which is the most efficient representation.  For heterogeneous collections (ports
with different concrete types, e.g. mixing `DeltaGapArrayPort` variants with different
excitation distributions, or mixing `DeltaGapPort` with `RectangularWaveguidePort`) `PT`
is the abstract supertype `PortType`, stored as a `Vector{PortType}`.

In both cases every consumer method dispatches on the *runtime* type of each element, so
behaviour is identical regardless of how the array was constructed.

# Fields
- `ports::Vector{PT}` -- Port vector (PT may be a concrete type or `PortType`)
- `numPorts::IT`       -- Number of ports
- `activePorts::Vector{IT}`  -- Indices of active (excited) ports
- `passivePorts::Vector{IT}` -- Indices of passive (load) ports

# Construction
```julia
# Homogeneous – all ports the same concrete type
pa = PortArray([port1, port2, port3])

# Heterogeneous – different DeltaGapArrayPort distribution types mixed freely
pa = PortArray([port_uniform, port_single_side, port_rectangular])

# Explicit heterogeneous helper
pa = PortArray{Float64, Int}([port_a, port_b])
```
"""
mutable struct PortArray{FT, IT, PT<:PortType}
    ports::Vector{PT}
    numPorts::IT
    activePorts::Vector{IT}
    passivePorts::Vector{IT}

    function PortArray{FT, IT, PT}(ports::Vector{PT}) where {FT<:Real, IT<:Integer, PT<:PortType}
        numPorts = length(ports)
        activePorts = IT[]
        passivePorts = IT[]

        for (idx, port) in enumerate(ports)
            if hasproperty(port, :isActive)
                if port.isActive
                    push!(activePorts, idx)
                else
                    push!(passivePorts, idx)
                end
            end
        end

        new(ports, numPorts, activePorts, passivePorts)
    end
end


# =============================================================================
# Private helper: extract (FT, IT) from any PortType instance
# =============================================================================

"""
    _port_numeric_types(port::PortType) -> (Type, Type)

Return the float precision type `FT` and integer type `IT` of a port by
inspecting its concrete type parameters.

All current port types follow the convention that the first type parameter is
`FT<:Real` and the second is `IT<:Integer`.  The function falls back to
`(Float64, Int)` when the parameters cannot be determined.
"""
function _port_numeric_types(port::PortType)
    T      = typeof(port)
    params = T.parameters
    FT = (length(params) >= 1 && params[1] isa Type && params[1] <: Real)    ? params[1] : Float64
    IT = (length(params) >= 2 && params[2] isa Type && params[2] <: Integer) ? params[2] : Int
    return FT, IT
end


# =============================================================================
# Outer constructors
# =============================================================================

"""
    PortArray(ports::Vector{PT}) where {PT<:PortType}

Construct a `PortArray` from a *homogeneous* vector whose element type is
already a single concrete `PortType`.  `FT` and `IT` are inferred from the
first element's type parameters.
"""
function PortArray(ports::Vector{PT}) where {PT<:PortType}
    isempty(ports) && error("PortArray requires at least one port")
    FT, IT = _port_numeric_types(ports[1])
    return PortArray{FT, IT, PT}(ports)
end

"""
    PortArray(ports::AbstractVector)

Flexible constructor that accepts *any* collection of `PortType` values,
including heterogeneous mixtures (e.g. `DeltaGapArrayPort` with different
excitation distribution type parameters, or different port kinds entirely).

All elements are promoted to a `Vector{PortType}` so that Julia's multiple
dispatch can handle each element at runtime with full type information.

`FT` and `IT` are inferred from the first element and validated to be
consistent across all ports.  An error is raised if any port uses a
different float or integer precision.

# Examples
```julia
# Mix DeltaGapArrayPort with UniformDistribution and SingleSideDistribution
pa = PortArray([
    DeltaGapArrayPort(; ..., excitationDistribution = UniformDistribution()),
    DeltaGapArrayPort(; ..., excitationDistribution = SingleSideDistribution(:left)),
])

# Mix different port kinds
pa = PortArray([delta_gap_port, rectangular_waveguide_port, current_probe_port])
```
"""
function PortArray(ports::AbstractVector)
    isempty(ports) && error("PortArray requires at least one port")
    all(p -> isa(p, PortType), ports) ||
        error("All elements must be subtypes of PortType")

    FT, IT = _port_numeric_types(ports[1])

    # Validate numeric type consistency across all ports
    for (i, port) in enumerate(ports)
        pFT, pIT = _port_numeric_types(port)
        pFT === FT || error(
            "Inconsistent float precision: port 1 uses $FT but port $i uses $pFT. " *
            "All ports in a PortArray must share the same FT and IT type parameters.")
        pIT === IT || error(
            "Inconsistent integer type: port 1 uses $IT but port $i uses $pIT. " *
            "All ports in a PortArray must share the same FT and IT type parameters.")
    end

    # Promote to abstract element type to allow heterogeneous storage
    port_vec = Vector{PortType}(ports)
    return PortArray{FT, IT, PortType}(port_vec)
end

"""
    PortArray{FT, IT}(ports::AbstractVector) where {FT<:Real, IT<:Integer}

Explicit floating-point / integer precision constructor for the heterogeneous case.
Useful when the element types cannot be inferred automatically.
"""
function PortArray{FT, IT}(ports::AbstractVector) where {FT<:Real, IT<:Integer}
    isempty(ports) && error("PortArray requires at least one port")
    all(p -> isa(p, PortType), ports) ||
        error("All elements must be subtypes of PortType")
    port_vec = Vector{PortType}(ports)
    return PortArray{FT, IT, PortType}(port_vec)
end


"""
    getExcitationVector(ports::PortArray, trianglesInfo::Vector{TriangleInfo{IT, FT}}, nbf::Integer, ieType::Symbol=:efie)

获取所有端口的总激励向量。

# 参数
- `ports`: 端口数组
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数
- `ieType::Symbol`: 积分方程类型 — `:efie` (默认), `:mfie`, `:cfie`

# 返回
- 总激励向量 V (复数数组)

# 注意
MFIE and CFIE excitation for ports are not fully implemented and will throw errors.
Use EFIE (default) for standard analysis.
"""
function getExcitationVector(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer,
    ieType::Symbol = :efie
) where {FT<:Real, IT<:Integer, PT<:PortType}

    # 初始化总激励向量
    V_total = zeros(Complex{FT}, nbf)

    # 遍历所有端口
    for port in ports.ports
        if ieType === :efie
            V_port = excitationVectorEFIE(port, trianglesInfo, nbf)
        elseif ieType === :mfie
            V_port = excitationVectorMFIE(port, trianglesInfo, nbf)
        elseif ieType === :cfie
            V_port = excitationVectorCFIE(port, trianglesInfo, nbf)
        else
            error("Unknown ieType: $ieType. Use :efie, :mfie, or :cfie")
        end
        V_total .+= V_port
    end

    return V_total
end


"""
    addExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    ieType::Symbol=:efie
)

将端口激励添加到已有向量中。

# 参数
- `V`: 预分配的激励向量 (会被修改)
- `ports`: 端口数组
- `trianglesInfo`: 三角形信息数组
- `ieType::Symbol`: 积分方程类型 — `:efie` (默认), `:mfie`, `:cfie`

# 注意
MFIE and CFIE excitation for ports are not fully implemented and will throw errors.
Use EFIE (default) for standard analysis.
"""
function addExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    ieType::Symbol = :efie
) where {FT<:Real, IT<:Integer, PT<:PortType}

    for port in ports.ports
        if ieType === :efie
            excitationVectorEFIE!(V, port, trianglesInfo)
        elseif ieType === :mfie
            # Note: excitationVectorMFIE! doesn't exist for ports, use non-mutating version
            V .+= excitationVectorMFIE(port, trianglesInfo, length(V))
        elseif ieType === :cfie
            # Note: excitationVectorCFIE! doesn't exist for ports, use non-mutating version
            V .+= excitationVectorCFIE(port, trianglesInfo, length(V))
        else
            error("Unknown ieType: $ieType. Use :efie, :mfie, or :cfie")
        end
    end

    return V
end



"""
    excitationVectorMFIE(
        ports::PortArray{FT, IT, PT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer;
        strategy::Symbol = :convert
    ) where {FT<:Real, IT<:Integer, PT<:PortType}

计算端口数组在MFIE方程中的总激励向量。
"""
function excitationVectorMFIE(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:PortType}

    V_total = zeros(Complex{FT}, nbf)

    for port in ports.ports
        if port isa CurrentProbe
            V_port = excitationVectorMFIE(port, trianglesInfo, nbf)
        else
            V_port = excitationVectorMFIE(port, trianglesInfo, nbf; strategy = strategy)
        end
        V_total .+= V_port
    end

    return V_total
end




"""
    excitationVectorCFIE(
        ports::PortArray{FT, IT, PT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer;
        alpha::FT = FT(0.5),
        mfie_strategy::Symbol = :convert
    ) where {FT<:Real, IT<:Integer, PT<:PortType}

计算端口数组在CFIE方程中的总激励向量。
"""
function excitationVectorCFIE(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    alpha::FT = FT(0.5),
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:PortType}

    V_total = zeros(Complex{FT}, nbf)

    for port in ports.ports
        if port isa CurrentProbe
            V_port = excitationVectorCFIE(port, trianglesInfo, nbf; alpha = alpha)
        else
            V_port = excitationVectorCFIE(port, trianglesInfo, nbf;
                                          alpha = alpha, mfie_strategy = mfie_strategy)
        end
        V_total .+= V_port
    end

    return V_total
end


# Assembly


# ============================================================
# 通用激励向量组装接口
# ============================================================

"""
    assembleExcitationVector!(
        V::Vector{Complex{FT}},
        ports::PortArray{FT, IT, PT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer,
        formulation::AbstractIntegralEquation
    ) where {FT<:Real, IT<:Integer, PT<:PortType}

统一的端口激励向量组装接口。

此函数根据所选的积分方程类型，自动分派到正确的激励向量计算方法。

# 参数
- `V`: 预分配的激励向量
- `ports`: 端口数组
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数
- `formulation`: 积分方程类型 (EFIE, MFIE, 或 CFIE)

# 示例
```julia
# EFIE
assembleExcitationVector!(V, ports, trianglesInfo, nbf, EFIE())

# MFIE
assembleExcitationVector!(V, ports, trianglesInfo, nbf, MFIE())

# CFIE with custom alpha
assembleExcitationVector!(V, ports, trianglesInfo, nbf, CFIE(alpha=0.3))
```
"""
function assembleExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer,
    formulation::EFIE
) where {FT<:Real, IT<:Integer, PT<:PortType}

    fill!(V, zero(Complex{FT}))
    return getExcitationVector(ports, trianglesInfo, nbf)
end


function assembleExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer,
    formulation::MFIE;
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:PortType}

    error("MFIE excitation for PortArray is not implemented. " *
          "Use EFIE excitation (industry standard) or implement a custom strategy if needed.")
end


function assembleExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer,
    formulation::CFIE{FT};
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:PortType}

    error("CFIE excitation for PortArray is not implemented. " *
          "Use EFIE excitation (industry standard) or implement a custom strategy if needed.")
end


"""
    getExcitationVector(
        ports::PortArray,
        trianglesInfo::Vector,
        nbf::Integer,
        formulation::AbstractIntegralEquation
    )

获取端口激励向量的便捷函数。

# 参数
- `ports`: 端口数组
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数
- `formulation`: 积分方程类型

# 返回
- 激励向量 V
"""
function getExcitationVector(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer,
    formulation::AbstractIntegralEquation
) where {FT<:Real, IT<:Integer, PT<:PortType}

    V = zeros(Complex{FT}, nbf)
    assembleExcitationVector!(V, ports, trianglesInfo, nbf, formulation)
    return V
end


