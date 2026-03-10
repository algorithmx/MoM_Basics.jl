"""
    DeltaGapPort{FT<:Real, IT<:Integer} <: ExcitingSource

Delta-Gap (电压间隙) 激励端口类型。

Delta-Gap 源是最常用的端口激励模型之一，它在导体表面或缝隙处模拟一个理想的电压源。
该模型假设在端口位置存在一个无限薄的电压间隙，间隙两侧存在电势差 V。

# 字段说明
```
id          ::IT                端口编号
V           ::Complex{FT}       端口激励电压 (复数形式，包含相位信息)
freq        ::FT                工作频率 (Hz)
portType    ::Symbol            端口类型
            :delta_gap         - Delta-Gap 电压源 (默认)
            :current_probe     - 电流探针激励
rwgID       ::IT               激励边所在的 RWG 基函数编号
triID_pos   ::IT               正基函数所在三角形编号
triID_neg   ::IT               负基函数所在三角形编号 (半基函数时为0)
edgel       ::FT               激励边边长
center      ::MVec3D{FT}       激励边中心位置 (全局坐标)
orient      ::MVec3D{FT}       端口方向 (沿边方向的单位向量)
isActive    ::Bool             是否为主动激励端口 (true) 或被动端口 (false)
```

# 物理背景

Delta-Gap 源模型基于以下假设：
1. 端口位置存在一个无限薄的理想电压间隙
2. 间隙两侧的电势差为 V (复数电压)
3. 该电压在基函数定义的边上产生切向电场

在 EFIE 方程中，电压激励向量 V 的计算公式为：
```
V_m = ∫ E_inc · f_m dS = V₀ × (l_m / 2)
```
其中 f_m 是第 m 个 RWG 基函数，l_m 是对应的边长。

# 构造函数

```julia
# 通过位置和方向创建端口 (自动寻找最近边)
DeltaGapPort(;
    id::IT = 0,
    V::Complex{FT} = one(Complex{FT}),
    freq::FT = Params.freq,
    position::MVec3D{FT} = zero(MVec3D{FT}),
    direction::MVec3D{FT} = zero(MVec3D{FT}),
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
)

# 直接通过基函数编号创建端口
DeltaGapPort(;
    id::IT = 0,
    V::Complex{FT} = one(Complex{FT}),
    freq::FT = Params.freq,
    rwgID::IT,
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
)
```

# 使用示例

```julia
# 方式1: 指定位置和方向，让程序自动寻找最近边
port = DeltaGapPort(
    id = 1,
    V = 1.0 + 0.0im,  # 1V 激励
    freq = 2.4e9,      # 2.4 GHz
    position = [0.0, 0.0, 0.0],  # 端口位置
    direction = [1.0, 0.0, 0.0],  # 端口方向
    trianglesInfo = trianglesInfo,
    rwgsInfo = rwgsInfo
)

# 方式2: 直接指定 RWG 基函数编号
port = DeltaGapPort(
    id = 1,
    V = 1.0 + 0.0im,
    freq = 2.4e9,
    rwgID = 42,  # 第42个RWG基函数
    trianglesInfo = trianglesInfo,
    rwgsInfo = rwgsInfo
)
```
"""
mutable struct DeltaGapPort{FT<:Real, IT<:Integer} <: ExcitingSource
    # 端口标识
    id          ::IT
    # 端口激励电压 (复数，包含幅值和相位)
    V           ::Complex{FT}
    # 工作频率
    freq        ::FT
    # 端口类型: :delta_gap, :current_probe
    portType    ::Symbol
    # RWG 基函数信息
    rwgID       ::IT           # 激励边所在的 RWG 基函数编号
    triID_pos   ::IT           # 正基函数所在三角形编号
    triID_neg   ::IT           # 负基函数所在三角形编号 (半基函数时为0)
    edgel       ::FT           # 激励边边长
    # 几何信息
    center      ::MVec3D{FT}   # 激励边中心位置 (全局坐标)
    orient      ::MVec3D{FT}   # 端口方向 (沿边方向的单位向量)
    # 端口属性
    isActive    ::Bool         # 是否为主动激励端口
end

"""
    DeltaGapPort{FT}(; kwargs...) where {FT<:Real}

类型自动转换的 `DeltaGapPort` 构造函数。

# 参数说明
- `id::Integer = 0`: 端口编号
- `V::Complex{FT} = one(Complex{FT})`: 端口电压 (默认1V)
- `freq::FT`: 工作频率 (Hz)
- `rwgID::Integer`: RWG基函数编号 (可选)
- `position::MVec3D{FT}`: 端口位置坐标 (可选)
- `direction::MVec3D{FT}`: 端口方向 (可选)
- `trianglesInfo`: 三角形信息数组
- `rwgsInfo`: RWG基函数信息数组
"""
function DeltaGapPort{FT, IT}(;
    id::IT = zero(IT),
    V::Complex{FT} = one(Complex{FT}),
    freq::FT = zero(FT),
    portType::Symbol = :delta_gap,
    rwgID::IT = zero(IT),
    triID_pos::IT = zero(IT),
    triID_neg::IT = zero(IT),
    edgel::FT = zero(FT),
    position::MVec3D{FT} = zero(MVec3D{FT}),
    direction::MVec3D{FT} = zero(MVec3D{FT}),
    trianglesInfo::Vector{TriangleInfo{IT, FT}} = TriangleInfo{IT, FT}[],
    rwgsInfo::Vector{RWG{IT, FT}} = RWG{IT, FT}[],
    isActive::Bool = true
) where {FT<:Real, IT<:Integer}

    # 复数类型
    CT = Complex{FT}

    # 如果提供了位置但没有指定rwgID，则需要搜索最近的边
    if rwgID == 0 && !iszero(position)
        # 调用边搜索函数
        result = find_edge_index(position, direction, trianglesInfo, rwgsInfo)
        rwgID = result.rwgID
        triID_pos = result.triID_pos
        triID_neg = result.triID_neg
        edgel = result.edgel
        position = result.center
        direction = result.orient
    end

    # 如果rwgID有效，获取基函数信息
    if rwgID > 0 && length(rwgsInfo) >= rwgID
        rwg = rwgsInfo[rwgID]
        edgel = rwg.edgel
        # 获取三角形信息
        triID_pos = rwg.inGeo[1]
        triID_neg = rwg.inGeo[2]
        # 计算边中心 (如果未提供)
        if iszero(position) && !iszero(triID_pos)
            position = rwg.center
        end
    end

    # 归一化方向向量
    dir_norm = norm(direction)
    if dir_norm > 0
        direction = direction ./ dir_norm
    end

    return DeltaGapPort{FT, IT}(
        id,
        CT(V),
        freq,
        portType,
        rwgID,
        triID_pos,
        triID_neg,
        edgel,
        position,
        direction,
        isActive
    )
end

"""
    DeltaGapPort(args...; kwargs...)

默认精度类型的 `DeltaGapPort` 构造函数。
"""
DeltaGapPort(args...; kwargs...) = DeltaGapPort{Precision.FT, IntDtype}(args...; kwargs...)


# ---------------------------------


"""
    sourceEfield(port::DeltaGapPort, r::AbstractVector{FT})

计算 Delta-Gap 端口在给定位置产生的入射电场。

对于 Delta-Gap 源，电场存在于端口位置的边上。
在理想情况下，Delta-Gap 源在边两侧产生均匀的电场：

```
E_inc = V / d × n̂
```

其中 V 是端口电压，d 是间隙宽度（理想化为0），n̂ 是边的法向。

# 注意
由于 Delta-Gap 源是线激励源，其"入射电场"通常指在基函数定义的边上
产生的切向电场分量，用于计算激励向量。
"""
function sourceEfield(port::DeltaGapPort{FT, IT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer}
    # Delta-Gap 源在边上有定义的电场
    # 这里返回零场，因为在实际激励向量计算中直接使用电压值
    return zero(MVec3D{Complex{FT}})
end


"""
    sourceHfield(port::DeltaGapPort, r::AbstractVector{FT})

计算 Delta-Gap 端口在给定位置产生的入射磁场。
"""
function sourceHfield(port::DeltaGapPort{FT, IT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer}
    return zero(MVec3D{Complex{FT}})
end


"""
    excitationVectorEFIE(port::DeltaGapPort, trianglesInfo::Vector{TriangleInfo{IT, FT}}, nbf::Integer)

计算 Delta-Gap 端口在 RWG 基函数上的激励向量。

# 物理公式
对于 Delta-Gap 激励，激励向量的计算公式为：

```
V_m = ∫ E_inc · f_m dS
```

其中：
- E_inc 是入射电场 (对于 Delta-Gap，是沿边方向的均匀场)
- f_m 是第 m 个 RWG 基函数
- 积分在三角形面上进行

对于 RWG 基函数，激励向量可以简化为：

```
V_m = V_0 × l_m / 2
```

其中 V_0 是端口电压，l_m 是第 m 条边的边长。

# 参数
- `port`: Delta-Gap 端口
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数

# 返回
- `V`: 激励向量 (复数数组，长度为 nbf)
"""
function excitationVectorEFIE(
    port::DeltaGapPort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer
) where {FT<:Real, IT<:Integer}

    # 预分配激励向量
    V = zeros(Complex{FT}, nbf)

    # 调用激励向量计算函数
    excitationVectorEFIE!(V, port, trianglesInfo)

    return V
end


"""
    excitationVectorEFIE!(V::Vector{Complex{FT}}, port::DeltaGapPort, trianglesInfo::Vector{TriangleInfo{IT, FT}})

将 Delta-Gap 端口的激励向量添加到已有向量 V 中。
"""
function excitationVectorEFIE!(
    V::Vector{Complex{FT}},
    port::DeltaGapPort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # 获取端口参数
    rwgID = port.rwgID
    portV = port.V
    edgel = port.edgel

    # Delta-Gap 激励直接应用到 RWG 基函数
    # V_m = V_0 × l_m / 2
    if rwgID > 0 && rwgID <= length(V)
        # 对于完整的 RWG 基函数 (有正负两边)
        if port.triID_neg > 0
            # 正基函数贡献: +V * l / 2
            V[rwgID] += portV * edgel / 2
        else
            # 半基函数 (边界元): 只考虑一边的贡献
            V[rwgID] += portV * edgel
        end
    end

    return V
end


"""
    getPortVoltage(port::DeltaGapPort, current::Complex{FT})

计算端口电压。

对于主动激励端口，返回设定的端口电压。
对于被动端口，可以从电流计算电压 (需要阻抗信息)。

# 参数
- `port`: 端口
- `current`: 端口电流

# 返回
- 端口电压 (复数)
"""
function getPortVoltage(port::DeltaGapPort{FT, IT}, current::Complex{FT}) where {FT<:Real, IT<:Integer}
    if port.isActive
        return port.V
    else
        # 对于被动端口，需要从电流计算
        # 这里返回设定电压，实际计算需要阻抗矩阵
        return port.V
    end
end


"""
    getPortCurrent(port::DeltaGapPort, voltage::Complex{FT} = port.V)

计算端口电流。

在已知电压和阻抗的情况下计算电流：
I = V / Z

# 参数
- `port`: 端口
- `voltage`: 端口电压 (可选，默认使用设定电压)

# 返回
- 端口电流 (复数)
"""
function getPortCurrent(port::DeltaGapPort{FT, IT}; Z::Complex{FT} = 50.0) where {FT<:Real, IT<:Integer}
    return port.V / Z
end



"""
    excitationVectorMFIE(
        port::DeltaGapPort{FT, IT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer;
        strategy::Symbol = :convert
    ) where {FT<:Real, IT<:Integer}

计算Delta-Gap端口在MFIE方程中的激励向量。

# 弃用警告
DEPRECATED: 此函数已弃用。assembleExcitationVector! 现在对 MFIE 使用 EFIE 激励向量。

# 参数
- `port`: Delta-Gap端口
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数
- `strategy::Symbol`: 激励策略
    - `:convert` (默认): 将电压源转换为等效电流源
    - `:hybrid`: 使用混合方法，在端口位置添加EFIE贡献

# 物理说明
MFIE天然使用磁场激励，而Delta-Gap是电压源（电场激励）。
由于物理上的不兼容性，需要进行转换处理。

# 转换策略 (:convert)
将电压源转换为等效电流源：
I_eq = V_port / Z_port

然后使用等效电流探针计算激励向量。

# 混合策略 (:hybrid)
在MFIE中对端口位置添加一个小的EFIE贡献，
以保持与物理直觉的一致性。
"""
function excitationVectorMFIE(
    port::DeltaGapPort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer}

    if strategy == :convert
        # 策略1：转换为等效电流源
        # I_eq = V / Z_port，假设端口阻抗为50欧姆
        Z_port = FT(50.0)  # 可配置
        I_eq = port.V / Z_port

        # 创建等效电流探针
        probe = CurrentProbe{FT, IT}(
            id = port.id,
            I = I_eq,
            freq = port.freq,
            rwgID = port.rwgID,
            triID = port.triID_pos,
            edgel = port.edgel,
            center = port.center,
            isActive = port.isActive
        )

        return excitationVectorMFIE(probe, trianglesInfo, nbf)

    elseif strategy == :hybrid
        # 策略2：混合方法
        # 在MFIE中使用简化的EFIE贡献（仅在端口附近）
        V = zeros(Complex{FT}, nbf)

        # 对端口所在的基函数，添加一个小的EFIE贡献
        if port.rwgID > 0 && port.rwgID <= nbf
            # 使用较小的系数避免破坏MFIE的数值特性
            epsilon = FT(1e-3)
            V[port.rwgID] = port.V * port.edgel * epsilon
        end

        return V
    else
        error("Unknown MFIE excitation strategy: $strategy")
    end
end



"""
    excitationVectorCFIE(
        port::DeltaGapPort{FT, IT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer;
        alpha::FT = FT(0.5),
        mfie_strategy::Symbol = :convert
    ) where {FT<:Real, IT<:Integer}

计算Delta-Gap端口在CFIE方程中的激励向量。

# 物理公式
CFIE是EFIE和MFIE的线性组合：
V_CFIE = α * V_EFIE + (1-α) * η * V_MFIE

其中：
- α (alpha) 是EFIE权重系数
- η (eta) 是自由空间阻抗（约377Ω）

# 参数
- `port`: Delta-Gap端口
- `trianglesInfo`: 三角形信息数组
- `nbf`: 基函数总数
- `alpha::FT`: EFIE权重系数 (默认0.5)
- `mfie_strategy::Symbol`: MFIE激励策略
"""
function excitationVectorCFIE(
    port::DeltaGapPort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    alpha::FT = FT(0.5),
    mfie_strategy::Symbol = :convert
) where {FT<:Real, IT<:Integer}

    # 获取EFIE激励向量
    V_efie = excitationVectorEFIE(port, trianglesInfo, nbf)

    # 获取MFIE激励向量
    V_mfie = excitationVectorMFIE(port, trianglesInfo, nbf; strategy = mfie_strategy)

    # 组合：V_CFIE = α * V_EFIE + (1-α) * η * V_MFIE
    eta = FT(eta_0)
    one_minus_alpha = FT(1.0) - alpha

    return alpha .* V_efie .+ one_minus_alpha .* eta .* V_mfie
end




# ============================================================
# S-参数计算函数
# ============================================================

"""
    computeInputImpedance(port::DeltaGapPort, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}}; Z0::FT = 50.0)

计算端口的输入阻抗。

# 物理原理
在MoM求解中，阻抗矩阵 Z 和激励向量 V 的关系为：
Z × I = V

其中 I 是电流系数向量。通过求解线性方程组得到 I 后，
可以计算端口处的输入阻抗：

Z_in = V_port / I_port

其中 V_port 是端口激励电压，I_port 是端口电流。

# 参数
- `port`: Delta-Gap 端口
- `Z_matrix`: MoM 阻抗矩阵 (nbf × nbf)
- `V_excitation`: 激励向量
- `Z0`: 参考阻抗 (默认 50 Ω)

# 返回
- 输入阻抗 Z_in (复数)
"""
function computeInputImpedance(
    port::DeltaGapPort{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer}

    nbf = size(Z_matrix, 1)

    # 验证端口编号有效
    if port.rwgID <= 0 || port.rwgID > nbf
        error("Invalid port rwgID: $(port.rwgID), must be between 1 and $nbf")
    end

    # 求解电流向量: Z × I = V
    # 使用 Julia 的内置线性代数求解器
    I = Z_matrix \ V_excitation

    # 获取端口处的电流（基函数系数）
    I_port = I[port.rwgID]

    # 避免除以零
    if abs(I_port) < eps(FT)
        error("Zero current at port $(port.id). Cannot compute input impedance.")
    end

    # 计算输入阻抗: Z_in = V_port / I_port
    Z_in = port.V / I_port

    return Z_in
end



"""
    computeS11(port::DeltaGapPort, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}}; Z0::FT = 50.0)

计算单端口的 S11 参数。

# 物理公式
S11 = (Z_in - Z0) / (Z_in + Z0)

其中：
- Z_in 是输入阻抗
- Z0 是参考阻抗（默认 50 Ω）

# 参数
- `port`: Delta-Gap 端口
- `Z_matrix`: MoM 阻抗矩阵
- `V_excitation`: 激励向量
- `Z0`: 参考阻抗

# 返回
- S11 复数值 (幅度和相位)
"""
function computeS11(
    port::DeltaGapPort{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer}

    # 计算输入阻抗
    Z_in = computeInputImpedance(port, Z_matrix, V_excitation; Z0 = Z0)

    # 计算 S11: (Z_in - Z0) / (Z_in + Z0)
    S11 = (Z_in - Z0) / (Z_in + Z0)

    return S11
end


"""
    getPortImpedance(port::DeltaGapPort, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}})

获取端口阻抗（输入阻抗的另一种说法）。

这是 computeInputImpedance 的便捷别名。
"""
function getPortImpedance(
    port::DeltaGapPort{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}}
) where {FT<:Real, IT<:Integer}

    return computeInputImpedance(port, Z_matrix, V_excitation)
end
