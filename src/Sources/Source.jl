## 本文件定义源类型和一些相关的计算函数
"""
激励源抽象类
"""
abstract type ExcitingSource end

"""
天线抽象类
"""
abstract type AntennaType <:ExcitingSource end


"""
端口抽象类
"""
abstract type PortType <:ExcitingSource end


"""
复合激励类
"""
const ExcitingSources = Union{ExcitingSource, AbstractVector{ExcitingSource}}

# 各种类型的源
# 平面波

"""
积分方程抽象类

不同的积分方程对同一端口激励可能有不同的处理方式
"""
abstract type AbstractIntegralEquation end

"""
    EFIE <: AbstractIntegralEquation

电场积分方程 (Electric Field Integral Equation)。

EFIE通过对导体表面施加电场边界条件来求解电流分布。

EFIE适用于开放结构如天线、散射体等。
"""
struct EFIE <: AbstractIntegralEquation
    function EFIE()
        return new()
    end
end

"""
    MFIE <: AbstractIntegralEquation

磁场积分方程 (Magnetic Field Integral Equation)。

MFIE通过对导体表面施加磁场边界条件来求解电流分布。

MFIE适用于封闭导体目标，具有较快的收敛速度。
"""
struct MFIE <: AbstractIntegralEquation
    function MFIE()
        return new()
    end
end

"""
    CFIE{FT} <: AbstractIntegralEquation

组合场积分方程 (Combined Field Integral Equation)。

CFIE是EFIE和MFIE的线性组合，兼具两者的优点。

CFIE同时施加电场和磁场边界条件，消除了各自的内谐振问题。

# 参数
- `alpha::FT`: EFIE权重系数，默认0.5
- `beta::FT`: MFIE权重系数，默认0.5
"""
struct CFIE{FT<:Real} <: AbstractIntegralEquation
    alpha::FT
    beta::FT
    function CFIE{FT}(; alpha::FT=FT(0.5), beta::FT=FT(0.5)) where FT<:Real
        @assert alpha + beta == 1 "CFIE权重系数之和必须为1"
        return new{FT}(alpha, beta)
    end
end

# 平面波
include("Planewave.jl")
# 磁偶极子
include("MagneticDipole.jl")
# 天线阵
include("AntettaArray.jl")
# 场提取
include("FieldExtraction.jl")

##