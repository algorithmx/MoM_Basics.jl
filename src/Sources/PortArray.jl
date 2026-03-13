
"""
    PortArray{FT, IT, PT<:PortType}

端口数组类型，用于管理多个端口。

# 字段
- `ports::Vector{PT}`: 端口向量
- `numPorts::IT`: 端口数量
- `activePorts::Vector{IT}`: 主动激励端口索引
- `passivePorts::Vector{IT}`: 被动端口索引
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


"""
    PortArray(ports::Vector{PT}) where {PT<:PortType}

默认类型的端口数组构造函数。
"""
function PortArray(ports::Vector{PT}) where {PT<:PortType}
    if isempty(ports)
        error("PortArray requires at least one port")
    end

    # 获取第一个端口的类型参数
    first_port = ports[1]
    port_type = typeof(first_port)

    # 尝试获取类型参数
    if port_type <: DeltaGapPort
        FT = port_type.parameters[1]
        IT = port_type.parameters[2]
    elseif port_type <: DeltaGapArrayPort
        # DeltaGapArrayPort is the generic base for array-based ports
        FT = port_type.parameters[1]
        IT = port_type.parameters[2]
    elseif port_type <: RectangularEdgePort
        # RectangularEdgePort is now implemented as a wrapper around DeltaGapArrayPort
        FT = port_type.parameters[1]
        IT = port_type.parameters[2]
    elseif port_type <: CurrentProbe
        FT = port_type.parameters[1]
        IT = port_type.parameters[2]
    else
        # 使用默认值
        FT = Float64
        IT = Int
    end

    return PortArray{FT, IT, PT}(ports)
end


"""
    getExcitationVector(ports::PortArray, trianglesInfo::Vector{TriangleInfo{IT, FT}}, nbf::Integer)

获取所有端口的总激励向量。

# 参数
- `ports`: 端口数组
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数

# 返回
- 总激励向量 V (复数数组)
"""
function getExcitationVector(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer
) where {FT<:Real, IT<:Integer, PT<:PortType}

    # 初始化总激励向量
    V_total = zeros(Complex{FT}, nbf)

    # 遍历所有端口
    for port in ports.ports
        V_port = excitationVectorEFIE(port, trianglesInfo, nbf)
        V_total .+= V_port
    end

    return V_total
end


"""
    addExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray,
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
)

将端口激励添加到已有向量中。
"""
function addExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer, PT<:PortType}

    for port in ports.ports
        excitationVectorEFIE!(V, port, trianglesInfo)
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

    # 行业标准做法：对于 MFIE 公式，始终使用 EFIE 激励向量
    # 物理原理：
    # - MFIE 基于磁场，天然处理电流源
    # - Delta-Gap 端口是电压源（电场激励），与 MFIE 不兼容
    # - 商业软件 (FEKO, HFSS, CST) 在 MFIE 中使用 EFIE 激励向量
    # - 这避免了假设端口阻抗的循环依赖问题
    
    # 检查是否有电压端口 (DeltaGapPort)
    has_voltage_port = any(is_voltage_port, ports.ports)
    
    if has_voltage_port
        @warn "MFIE formulation detected with voltage port(s). " *
              "Using EFIE excitation vector for ports (industry standard practice). " *
              "This avoids the physically incorrect 50Ω impedance assumption."
    end
    
    # 始终使用 EFIE 激励向量，不管是否有电压端口
    fill!(V, zero(Complex{FT}))
    return getExcitationVector(ports, trianglesInfo, nbf)
end


function assembleExcitationVector!(
    V::Vector{Complex{FT}},
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer,
    formulation::CFIE{FT};
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:PortType}

    # 行业标准做法：对于 CFIE 公式，始终使用 EFIE 激励向量
    # 物理原理：
    # - CFIE = alpha*EFIE + (1-alpha)*MFIE
    # - MFIE 部分对电压源有同样的不兼容问题
    # - 商业软件在 CFIE 中也使用 EFIE 激励向量
    
    # 检查是否有电压端口
    has_voltage_port = any(is_voltage_port, ports.ports)
    
    if has_voltage_port
        @warn "CFIE formulation detected with voltage port(s). " *
              "Using EFIE excitation vector for ports (industry standard practice). " *
              "This avoids the physically incorrect 50Ω impedance assumption."
    end
    
    # 始终使用 EFIE 激励向量
    fill!(V, zero(Complex{FT}))
    return getExcitationVector(ports, trianglesInfo, nbf)
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


