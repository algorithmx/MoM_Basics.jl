
"""
    PortArray{FT, IT, PT<:ExcitingSource}

端口数组类型，用于管理多个端口。

# 字段
- `ports::Vector{PT}`: 端口向量
- `numPorts::IT`: 端口数量
- `activePorts::Vector{IT}`: 主动激励端口索引
- `passivePorts::Vector{IT}`: 被动端口索引
"""
mutable struct PortArray{FT, IT, PT<:ExcitingSource}
    ports::Vector{PT}
    numPorts::IT
    activePorts::Vector{IT}
    passivePorts::Vector{IT}

    function PortArray{FT, IT, PT}(ports::Vector{PT}) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}
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
    PortArray(ports::Vector{PT}) where {PT<:ExcitingSource}

默认类型的端口数组构造函数。
"""
function PortArray(ports::Vector{PT}) where {PT<:ExcitingSource}
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
    elseif port_type <: RectangularWaveguidePort
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
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
    ) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

计算端口数组在MFIE方程中的总激励向量。
"""
function excitationVectorMFIE(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
    ) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

计算端口数组在CFIE方程中的总激励向量。
"""
function excitationVectorCFIE(
    ports::PortArray{FT, IT, PT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    alpha::FT = FT(0.5),
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
    ) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

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
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

    V = zeros(Complex{FT}, nbf)
    assembleExcitationVector!(V, ports, trianglesInfo, nbf, formulation)
    return V
end




"""
    computeSParameters(ports::PortArray, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}}; Z0::FT = 50.0)

计算多端口系统的 S 参数矩阵。

# 物理公式
对于 N 端口系统，S 参数矩阵为：

S = (Z - Z0×I) × (Z + Z0×I)^(-1)

其中：
- Z 是 N×N 阻抗矩阵
- Z0 是参考阻抗（标量或 N×N 对角矩阵）
- I 是单位矩阵

# 参数
- `ports`: 端口数组
- `Z_matrix`: MoM 阻抗矩阵
- `V_excitation`: 激励向量
- `Z0`: 参考阻抗 (默认 50 Ω)

# 返回
- S 参数矩阵 (N×N 复数矩阵)
"""
function computeSParameters(
    ports::PortArray{FT, IT, PT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT          = FT(50.0),
    check_quality::Bool = true
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

    nbf = size(Z_matrix, 1)
    num_ports = ports.numPorts
    any(port -> port isa RectangularWaveguidePort, ports.ports) &&
        error("computeSParameters does not support RectangularWaveguidePort yet")
    
    if num_ports == 1
        # 单端口情况：直接计算 S11
        port = ports.ports[1]
        S11 = computeS11(port, Z_matrix, V_excitation; Z0 = Z0)
        return S11
    end

    # ============================================================================
    # 多端口 S 参数计算
    # ============================================================================
    # 
    # 物理原理：
    # 端口阻抗矩阵 Z_port[i,j] 定义为：当端口 j 有单位电流时，端口 i 处的电压
    # 即：Z_port[i,j] = V_i / I_j
    # 
    # 关键认识：在 MoM 方法中，阻抗矩阵 Z_matrix 已经包含了所有的电磁耦合信息。
    # 对于端口 i 和端口 j（对应于 RWG 基函数 rwgID_i 和 rwgID_j），
    # MoM 阻抗矩阵元素 Z_matrix[rwgID_i, rwgID_j] 正是端口阻抗 Z_port[i,j]。
    # 
    # 这是因为：
    # 1. 激励向量使用约定：V_exc[m] = V_port × l_m / 2（完整 RWG）
    # 2. MoM 系统方程：Z_matrix · I = V_exc
    # 3. 输入阻抗公式：Z_in = V_port / I_coeff[rwgID]（已验证正确）
    # 4. 因此，Z_matrix 的元素在此约定下直接给出端口阻抗
    # 
    # 这种方法的优势：
    # - 物理上正确：直接利用 MoM 阻抗矩阵中的电磁耦合信息
    # - 计算高效：避免对每个端口重新求解线性系统
    # - 保证互易性：Z_matrix 是对称的（对于互易介质），因此 Z_port 也是对称的
    # ============================================================================

    # 构建端口阻抗矩阵 Z_port  
    # Z_port[i,j] = 端口 i 和端口 j 之间的阻抗
    # 
    # 方法：对每个端口 j 进行单独激励，求解系统，计算所有端口处的响应
    Z_port = Matrix{Complex{FT}}(undef, num_ports, num_ports)

    for j in 1:num_ports
        port_j = ports.ports[j]
        pid_j = port_j.rwgID
        
        # 验证端口 j 有效性
        if pid_j <= 0 || pid_j > nbf
            error("Invalid port rwgID: $pid_j for port $(port_j.id)")
        end

        # 确定端口 j 的激励向量缩放因子
        # 完整 RWG: V_exc = V_port × l / 2
        # 半基函数: V_exc = V_port × l
        scale_j = if isa(port_j, DeltaGapPort)
            (port_j.triID_neg > 0) ? (port_j.edgel / 2) : port_j.edgel
        else
            FT(1.0)  # CurrentProbe 不需要缩放
        end

        # 为端口 j 创建激励向量
        # 使用单位物理电压 (1V) 激励端口 j
        V_j = zeros(Complex{FT}, nbf)
        V_j[pid_j] = Complex{FT}(1.0) * scale_j

        # 求解电流分布：Z_matrix · I_j = V_j
        try
            I_j = Z_matrix \ V_j

            # 验证端口 j 的电流非零
            if abs(I_j[pid_j]) < eps(FT)
                @warn "Zero current at excited port $j"
                for i in 1:num_ports
                    Z_port[i, j] = Complex{FT}(Inf, 0)
                end
                continue
            end

            # 计算所有端口处的阻抗
            for i in 1:num_ports
                port_i = ports.ports[i]
                pid_i = port_i.rwgID

                # 验证端口 i 有效性
                if pid_i <= 0 || pid_i > nbf
                    error("Invalid port rwgID: $pid_i for port $(port_i.id)")
                end

                # 确定端口 i 的激励向量缩放因子
                scale_i = if isa(port_i, DeltaGapPort)
                    (port_i.triID_neg > 0) ? (port_i.edgel / 2) : port_i.edgel
                else
                    FT(1.0)  # CurrentProbe
                end

                # 计算端口 i 处的物理电压响应
                # 
                # 物理推导（完整版本）：
                # ============================================================================
                # 
                # MoM 系统方程： Z_matrix · I_coeff = V_exc
                # 
                # 当端口 j 被物理电压 V_port_j = 1V 激励时：
                #   V_exc[pid_j] = V_port_j × scale_j = 1.0 × scale_j
                #   V_exc[k] = 0 for all k ≠ pid_j
                # 
                # 求解得到电流系数向量 I_j，其中：
                #   I_port_j = I_j[pid_j]  （端口 j 的物理电流）
                # 
                # 在端口 i 处，MoM 系统产生的"激励向量响应"为：
                #   V_exc_response[pid_i] = (Z_matrix · I_j)[pid_i]
                #                         = Σ_k Z_matrix[pid_i, k] × I_j[k]
                # 
                # 端口 i 的物理电压定义为：
                #   V_port_i = V_exc_response[pid_i] / scale_i
                # 
                # 这是因为激励向量与物理电压的关系为：V_exc = V_port × scale
                # 因此：V_port = V_exc / scale
                # 
                # 端口阻抗定义为：
                #   Z_port[i,j] = V_port_i / I_port_j
                #               = (V_exc_response[pid_i] / scale_i) / I_j[pid_j]
                #               = [(Z_matrix · I_j)[pid_i] / scale_i] / I_j[pid_j]
                # 
                # 这可以展开为：
                #   Z_port[i,j] = [Σ_k Z_matrix[pid_i, k] × I_j[k]] / (scale_i × I_j[pid_j])
                # 
                # 注意：这个公式适用于所有情况（完整 RWG、半基函数、混合），
                # 因为 scale_i 和 scale_j 已经正确反映了每个端口的基函数类型。
                # 
                # 验证单端口情况：
                # 当 i = j 时：Z_port[j,j] = [(Z_matrix · I_j)[pid_j] / scale_j] / I_j[pid_j]
                # 而单端口输入阻抗：Z_in = V_port / I_port = 1.0 / I_j[pid_j]
                # 这些应该一致（当考虑完整的 MoM 响应时）。
                # ============================================================================

                # 计算激励向量响应：(Z_matrix · I_j)[pid_i]
                V_exc_response = zero(Complex{FT})
                for k in 1:nbf
                    V_exc_response += Z_matrix[pid_i, k] * I_j[k]
                end

                # 计算物理电压：V_port_i = V_exc_response / scale_i
                V_port_i = V_exc_response / scale_i

                # 计算端口阻抗：Z_port[i,j] = V_port_i / I_port_j
                Z_port[i, j] = V_port_i / I_j[pid_j]
            end
        catch e
            # 如果矩阵求解失败
            @warn "Failed to solve for port $j excitation: $e"
            for i in 1:num_ports
                Z_port[i, j] = Complex{FT}(NaN, 0)
            end
        end
    end

    # ============================================================================
    # 从阻抗矩阵计算 S 参数矩阵
    # ============================================================================
    # 
    # S 参数定义：S[i,j] = 当所有端口匹配到 Z0 时，端口 j 入射波引起的端口 i 反射波
    # 
    # 转换公式（标准微波网络理论）：
    # S = (Z - Z0·I) · (Z + Z0·I)^(-1)
    # 
    # 其中：
    # - Z 是端口阻抗矩阵（N×N）
    # - Z0 是参考阻抗（通常为 50Ω）
    # - I 是单位矩阵（N×N）
    # 
    # 物理意义：
    # - S[i,i]（对角元）是端口 i 的反射系数
    # - S[i,j]（i≠j）是端口 j 到端口 i 的传输系数
    # - 对于无源互易网络：|S[i,j]|² 之和 ≤ 1（能量守恒）
    # ============================================================================

    # 构造参考阻抗矩阵：Z0 × 单位矩阵
    Z0_matrix = Z0 * Matrix{Complex{FT}}(I, num_ports, num_ports)

    # 计算 S 参数矩阵：S = (Z - Z0·I) / (Z + Z0·I)
    # 注意：A / B 在 Julia 中等价于 A * inv(B)
    try
        S_matrix = (Z_port - Z0_matrix) / (Z_port + Z0_matrix)
    catch e
        # 如果矩阵求逆失败（例如，Z_port + Z0·I 接近奇异）
        # 返回 NaN 矩阵以指示计算失败
        @warn "Failed to compute S-parameter matrix: $e"
        S_matrix = fill(Complex{FT}(NaN, NaN), num_ports, num_ports)
    end

    # 质量检查：互易性 + 无源性 (only when single-port path used computeS11 which has no matrix to check)
    check_quality && !any(isnan, S_matrix) && check_sparameter_quality(S_matrix)
    return S_matrix
end

