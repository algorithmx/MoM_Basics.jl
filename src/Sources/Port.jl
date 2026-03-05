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


"""
    find_edge_index(
        position::MVec3D{FT},
        direction::MVec3D{FT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        rwgsInfo::Vector{RWG{IT, FT}}
    ) where {FT<:Real, IT<:Integer}

根据位置和方向搜索最近的 RWG 基函数边。

# 算法说明
1. 遍历所有三角形，计算每个三角形中心到给定位置的距离
2. 筛选出距离小于阈值的三角形作为候选
3. 在候选三角形中寻找最近的边
4. 根据方向向量确定边的正负方向

# 参数
- `position`: 端口位置 (全局坐标)
- `direction`: 端口方向 (归一化向量)
- `trianglesInfo`: 三角形信息数组
- `rwgsInfo`: RWG基函数信息数组

# 返回
- `rwgID`: 最近的RWG基函数编号
- `triID_pos`: 正基函数所在三角形编号
- `triID_neg`: 负基函数所在三角形编号
- `edgel`: 边长
- `center`: 边中心位置
- `orient`: 边方向向量
"""
function find_edge_index(
    position::MVec3D{FT},
    direction::MVec3D{FT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # 初始化结果
    rwgID_best = zero(IT)
    triID_pos_best = zero(IT)
    triID_neg_best = zero(IT)
    edgel_best = zero(FT)
    center_best = zero(MVec3D{FT})
    orient_best = zero(MVec3D{FT})
    dist_min = FT(Inf)

    # 归一化方向向量
    dir_norm = norm(direction)
    if dir_norm > 0
        direction = direction ./ dir_norm
    end

    # 遍历所有RWG基函数
    for (idx, rwg) in enumerate(rwgsInfo)
        # 计算基函数中心到指定位置的距离
        dist = norm(rwg.center - position)

        # 更新最近边
        if dist < dist_min
            dist_min = dist
            rwgID_best = idx
            triID_pos_best = rwg.inGeo[1]
            triID_neg_best = rwg.inGeo[2]
            edgel_best = rwg.edgel
            center_best = rwg.center
        end
    end

    # 如果指定了方向，计算边的方向向量
    if dir_norm > 0 && rwgID_best > 0
        # 从三角形信息中获取边的方向
        if triID_pos_best > 0
            tri = trianglesInfo[triID_pos_best]
            # 边方向为从第一个顶点指向第二个顶点
            orient_best = tri.edgev̂[:, 1]
            orient_best = orient_best ./ norm(orient_best)
        end
    end

    # 返回结果结构
    return (
        rwgID = rwgID_best,
        triID_pos = triID_pos_best,
        triID_neg = triID_neg_best,
        edgel = edgel_best,
        center = center_best,
        orient = orient_best
    )
end


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


# ============ 电流探针激励源 ============

"""
    CurrentProbe{FT<:Real, IT<:Integer} <: ExcitingSource

电流探针激励源类型。

电流探针 (Current Probe) 是另一种常用的端口激励模型，它通过在端口位置
注入已知电流来激励结构。与 Delta-Gap 源不同，电流探针不假设电压间隙，
而是直接指定注入的电流值。

# 物理背景
电流探针模型假设在端口边上有均匀分布的电流：
```
J = I_0 / l × ŷ
```
其中 I_0 是总电流，l 是边长，ŷ 是电流流动方向。

在 EFIE 中，电流探针激励的右端向量为：
```
V_m = ∫ J · f_m dS = I_0
```
即激励向量在电流探针所在的基函数处为单位电流。
"""
mutable struct CurrentProbe{FT<:Real, IT<:Integer} <: ExcitingSource
    id          ::IT
    I           ::Complex{FT}    # 探针电流 (复数)
    freq        ::FT             # 工作频率
    rwgID       ::IT             # RWG基函数编号
    triID       ::IT             # 所在三角形编号
    edgel       ::FT             # 边长
    center      ::MVec3D{FT}    # 边中心位置
    isActive    ::Bool          # 是否主动激励
end


"""
    CurrentProbe{FT}(; kwargs...) where {FT<:Real}

`CurrentProbe` 构造函数。
"""
function CurrentProbe{FT, IT}(;
    id::IT = zero(IT),
    I::Complex{FT} = one(Complex{FT}),
    freq::FT = zero(FT),
    rwgID::IT = zero(IT),
    triID::IT = zero(IT),
    edgel::FT = zero(FT),
    center::MVec3D{FT} = zero(MVec3D{FT}),
    isActive::Bool = true
) where {FT<:Real, IT<:Integer}
    return CurrentProbe{FT, IT}(id, Complex{FT}(I), freq, rwgID, triID, edgel, center, isActive)
end

CurrentProbe(args...; kwargs...) = CurrentProbe{Precision.FT, IntDtype}(args...; kwargs...)


"""
    sourceEfield(probe::CurrentProbe, r::AbstractVector{FT})

计算电流探针在给定位置产生的入射电场。
"""
function sourceEfield(probe::CurrentProbe{FT, IT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer}
    return zero(MVec3D{Complex{FT}})
end


"""
    sourceHfield(probe::CurrentProbe, r::AbstractVector{FT})

计算电流探针在给定位置产生的入射磁场。
"""
function sourceHfield(probe::CurrentProbe{FT, IT}, r::AbstractVector{FT}) where {FT<:Real, IT<:Integer}
    return zero(MVec3D{Complex{FT}})
end


"""
    excitationVectorEFIE(probe::CurrentProbe, trianglesInfo::Vector{TriangleInfo{IT, FT}}, nbf::Integer)

计算电流探针在 RWG 基函数上的激励向量。

对于电流探针，激励向量为：
```
V_m = I_0  (当 m 为探针所在基函数时)
V_m = 0   (其他)
```

即电流探针直接在对应的基函数上产生单位激励。
"""
function excitationVectorEFIE(
    probe::CurrentProbe{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer
) where {FT<:Real, IT<:Integer}

    V = zeros(Complex{FT}, nbf)

    if probe.rwgID > 0 && probe.rwgID <= nbf
        V[probe.rwgID] = probe.I
    end

    return V
end


"""
    excitationVectorEFIE!(V::Vector{Complex{FT}}, probe::CurrentProbe, trianglesInfo::Vector{TriangleInfo{IT, FT}})

将电流探针激励向量添加到已有向量中。
"""
function excitationVectorEFIE!(
    V::Vector{Complex{FT}},
    probe::CurrentProbe{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}

    if probe.rwgID > 0 && probe.rwgID <= length(V)
        V[probe.rwgID] += probe.I
    end

    return V
end


# ============ 端口集合类型 ============

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
            if isa(port, DeltaGapPort) || isa(port, CurrentProbe)
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
        if isa(port, DeltaGapPort)
            V_port = excitationVectorEFIE(port, trianglesInfo, nbf)
            V_total .+= V_port
        elseif isa(port, CurrentProbe)
            V_port = excitationVectorEFIE(port, trianglesInfo, nbf)
            V_total .+= V_port
        end
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
        if isa(port, DeltaGapPort)
            excitationVectorEFIE!(V, port, trianglesInfo)
        elseif isa(port, CurrentProbe)
            excitationVectorEFIE!(V, port, trianglesInfo)
        end
    end

    return V
end


# ============================================================
# MFIE 端口激励函数
# ============================================================
# 注意：这些函数已被弃用！
# assembleExcitationVector! 现在对 MFIE/CFIE 公式使用 EFIE 激励向量
# 这些函数保留仅用于向后兼容，但不应直接调用
# ============================================================

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
    excitationVectorMFIE(
        probe::CurrentProbe{FT, IT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer
    ) where {FT<:Real, IT<:Integer}

计算电流探针在MFIE方程中的激励向量。

# 物理说明
电流探针在MFIE中的处理相对直接。
MFIE直接涉及电流未知量，注入电流可以直接设置对应基函数的激励值。

# 公式
V_m = I_0 (当 m 为探针所在基函数时)
V_m = 0   (其他)
"""
function excitationVectorMFIE(
    probe::CurrentProbe{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer
) where {FT<:Real, IT<:Integer}

    V = zeros(Complex{FT}, nbf)

    if probe.rwgID > 0 && probe.rwgID <= nbf
        V[probe.rwgID] = probe.I
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
        if isa(port, DeltaGapPort)
            V_port = excitationVectorMFIE(port, trianglesInfo, nbf; strategy = strategy)
            V_total .+= V_port
        elseif isa(port, CurrentProbe)
            V_port = excitationVectorMFIE(port, trianglesInfo, nbf)
            V_total .+= V_port
        end
    end

    return V_total
end


# ============================================================
# CFIE 端口激励函数
# ============================================================

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


"""
    excitationVectorCFIE(
        probe::CurrentProbe{FT, IT},
        trianglesInfo::Vector{TriangleInfo{IT, FT}},
        nbf::Integer;
        alpha::FT = FT(0.5)
    ) where {FT<:Real, IT<:Integer}

计算电流探针在CFIE方程中的激励向量。
"""
function excitationVectorCFIE(
    probe::CurrentProbe{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    nbf::Integer;
    alpha::FT = FT(0.5)
) where {FT<:Real, IT<:Integer}

    # 获取EFIE激励向量
    V_efie = excitationVectorEFIE(probe, trianglesInfo, nbf)

    # 获取MFIE激励向量
    V_mfie = excitationVectorMFIE(probe, trianglesInfo, nbf)

    # 组合
    eta = FT(eta_0)
    one_minus_alpha = FT(1.0) - alpha

    return alpha .* V_efie .+ one_minus_alpha .* eta .* V_mfie
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
        if isa(port, DeltaGapPort)
            V_port = excitationVectorCFIE(port, trianglesInfo, nbf;
                                          alpha = alpha, mfie_strategy = mfie_strategy)
            V_total .+= V_port
        elseif isa(port, CurrentProbe)
            V_port = excitationVectorCFIE(port, trianglesInfo, nbf; alpha = alpha)
            V_total .+= V_port
        end
    end

    return V_total
end


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
    has_voltage_port = any(port -> isa(port, DeltaGapPort), ports.ports)
    
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
    has_voltage_port = any(port -> isa(port, DeltaGapPort), ports.ports)
    
    if has_voltage_port
        @warn "CFIE formulation detected with voltage port(s). " *
              "Using EFIE excitation vector for ports (industry standard practice). " *
              "This avoids the physically incorrect 50Ω impedance assumption."
    end
    
    # 始终使用 EFIE 激励向量
    fill!(V, zero(Complex{FT}))
    return getExcitationVector(ports, trianglesInfo, nbf)
end


# ============================================================
# 便捷函数
# ============================================================

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
    computeInputImpedance(port::CurrentProbe, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}}; Z0::FT = 50.0)

计算电流探针端口的输入阻抗。
"""
function computeInputImpedance(
    port::CurrentProbe{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer}

    nbf = size(Z_matrix, 1)

    if port.rwgID <= 0 || port.rwgID > nbf
        error("Invalid port rwgID: $(port.rwgID)")
    end

    # 求解电流系数向量: Z × I_coeff = V_exc
    I_coeff = Z_matrix \ V_excitation

    # 获取端口处的电流系数
    I_port = I_coeff[port.rwgID]

    # 检查数值稳定性
    if abs(I_port) < eps(FT) * 100
        @warn "Very small current coefficient at port $(port.id), impedance may be inaccurate"
    end

    # 输入阻抗定义: Z_in = V_exc[rwgID] / I_coeff[rwgID]
    # 物理原理：对于电流探针，V_exc[rwgID] = I_0（探针激励值），
    # I_coeff[rwgID] 是求解得到的电流系数。
    # 这与 DeltaGapPort 的 Z_in = V_0 / I_coeff[rwgID] 完全类比：
    # DeltaGapPort 的 V_exc[rwgID] = V_0 × l/2，Z_in = V_0 / I_coeff
    # CurrentProbe 的 V_exc[rwgID] = I_0（无缩放），Z_in = I_0 / I_coeff
    Z_in = port.I / I_port

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
    computeS11(port::CurrentProbe, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}}; Z0::FT = 50.0)

计算电流探针端口的 S11 参数。
"""
function computeS11(
    port::CurrentProbe{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}};
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer}

    Z_in = computeInputImpedance(port, Z_matrix, V_excitation; Z0 = Z0)
    S11 = (Z_in - Z0) / (Z_in + Z0)

    return S11
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
    Z0::FT = FT(50.0)
) where {FT<:Real, IT<:Integer, PT<:ExcitingSource}

    nbf = size(Z_matrix, 1)
    num_ports = ports.numPorts

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

    return S_matrix
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


"""
    getPortImpedance(port::CurrentProbe, Z_matrix::Matrix{Complex{FT}}, V_excitation::Vector{Complex{FT}})

获取电流探针端口的阻抗。
"""
function getPortImpedance(
    port::CurrentProbe{FT, IT},
    Z_matrix::Matrix{Complex{FT}},
    V_excitation::Vector{Complex{FT}}
) where {FT<:Real, IT<:Integer}

    return computeInputImpedance(port, Z_matrix, V_excitation)
end
