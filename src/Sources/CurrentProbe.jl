
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


# Excitation vector functions are implemented in MoM_Kernels.jl (SurfacePortExcitation.jl)

