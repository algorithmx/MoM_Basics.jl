
"""
    CurrentProbe{FT<:Real, IT<:Integer} <: PortType

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
mutable struct CurrentProbe{FT<:Real, IT<:Integer} <: PortType
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


