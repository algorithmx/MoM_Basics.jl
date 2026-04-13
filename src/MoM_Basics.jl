module MoM_Basics

using Dates, ProgressMeter, Printf, Reexport
using StaticArrays, OffsetArrays, SparseArrays
using LinearAlgebra, FastGaussQuadrature, Statistics
using .Threads, ThreadsX
using Rotations
using NPZ

export  Vec3D, SVec3D, MVec3D, random_rhat,
        ∠Info, θϕInfo, r̂func, θhatfunc, ϕhatfunc, r̂θϕInfo,
        θϕInfofromCart, nodes2Poles,
        globalObs2LocalObs, localObs2GlobalObs,
        globalrvec2Local, localrvec2Global,
        dist, greenfunc,
        VecCart2SphereMat, cart2sphereMat, cart2sphere,
        eulerZunit, eulerRotationMat, eulerRMat2αβγ,
        sincmath, sphere2cart,
        IntDtype, Precision, 
        Params, modiParams!, setPrecision!, setRecordMem!,
        inputBasicParameters, saveSimulationParams,
        C_0, μ_0, ε_0, η_0, div4π, ηdiv16π,
        GaussQuadratureInfo,
        SimulationParamsType, SimulationParams, modiSimulationParams!,
        VSBFTypes, updateVSBFTypes!, updateVSBFTParams!,
        VSCellType, SurfaceCellType, VolumeCellType,
        TriangleMesh, TriangleInfo, GQPNTri, GQPNTriSglr, 
        TriGQInfo, TriGQInfoSglr, getGQPTri, getGQPTriSglr, 
        setTricoor!, setTriParam!, getGQPTri, getGQPTriSglr,
        TetrahedraInfo, GQPNTetra, GQPNTetraSglr, Tris4Tetra,
        TetraGQInfo, TetraGQInfoSglr, setTetraCoor!, setTetraParam!,
        getTetrasInfo, setδκ!, getGQPTetra, getGQPTetraSglr, 
        HexahedraMesh, Quads4Hexa, GQPNQuad1D, GQPNQuad1DSglr, GQPNQuad1DSSglr, 
        GQPNQuad, GQPNQuadSglr, GQPNQuadSSglr, QuadGQInfo, QuadGQInfoSglr, QuadGQInfoSSglr,
        getGQPQuad, getGQPQuadSglr, getGQPQuadSSglr, Quads4Hexa,
        HexahedraInfo, GQPNHexa, GQPNHexaSglr, GQPNHexaSSglr,
        HexaGQInfo, HexaGQInfoSglr, HexaGQInfoSSglr,
        setHexaCoor!, setHexaParam!, getHexasInfo,
        getGQPHexa, getGQPHexaSglr, getGQPHexaSSglr, 
        setGeosPermittivity!,
        MeshDataType, getNodeElems, 
        getCellsBFs, getCellsFromFileName,
        getCellsBFsFromFileName, getBFsFromMeshData, getMeshData, getConnectionMatrix,
        RWG, PWC, SWG, RBF, LinearBasisFunction, ConstBasisFunction, BasisFunctionType,
        ExcitingSource, PortType, AntennaType, ExcitingSources,
        calExcitationFields, saveExcitationFields, ExcitationFieldData,
        calIncidentFields, saveIncidentFields, saveFieldData, mergeFieldData!, FieldData,
        triangleConnectivity,
        PlaneWave, sourceEfield, sourceHfield, sourceLocalEfield,
        MagneticDipole, update_phase!, add_phase!, update_orient!,
        sourceLocalFarEfield, sourceFarEfield, radiationIntegralL0, radiationIntensityU_m,
        radiationPower, radiationDirectionCoeff,
        # 端口激励
        DeltaGapPort, RectangularEdgePort, DeltaGapArrayPort, 
        CircularDeltaGapPort, EllipticalDeltaGapPort,
        CurrentProbe, PortArray, find_edge_index,
        bind_to_mesh!, unbind_mesh!,
        set_excitation_mode!, set_excitation_distribution!, get_edge_info, get_port_power,
        excitationVectorEFIE, excitationVectorEFIE!, excitationVectorMFIE, excitationVectorCFIE,
        getExcitationVector, addExcitationVector!,
        getPortVoltage, getPortCurrent, assembleExcitationVector!,
        # 激励分布类型
        AbstractExcitationDistribution, UniformDistribution, SingleSideDistribution, CustomDistribution,
        LeftSideDistribution, RightSideDistribution, BottomSideDistribution, TopSideDistribution,
        compute_voltage,
        # 模式阻抗计算
        compute_mode_cutoff, compute_mode_impedance,
        # 计算质量检查
        check_impedance_symmetry, check_sparameter_reciprocity,
        check_passivity, check_sparameter_quality,
        # 积分方程类型
        AbstractIntegralEquation, EFIE, MFIE, CFIE,
        AbstractAntennaArray, taylorwin, AntennaArray, distance,
        antennaArray, setdiffArray!,
        timer, memory, @clock, show_memory_time,
        # Green's function types and configuration (new for layered media support)
        AbstractGreenFunction, FreeSpaceGF, GroundPlaneGF, LayeredMediaGF,
        GreenFuncVals, evaluate_greenfunc, evaluate_greenfunc_A, evaluate_greenfunc_phi,
        create_green_function, set_layer_stack!, horizontal_distance, mirror_point_across_ground,
        LayerStack, LayerInfo, DCIMCoefficients, ComplexImage


## 记录程序运行内存占用情况
include("Recorder.jl")

## 各部分函数
# 网格元高斯求积点、权重计算函数
include("GaussQuadrature4Geos.jl")
@reexport using .GaussQuadrature4Geo

# 一些重要的要用到的基础类定义
include("BasicStuff.jl")
include("CoorTrans.jl")
# 一些有用的函数
include("UsefulFunctions.jl")

# 参数
include("ParametersSet.jl")
# 参数输入输出
include("Inputs.jl")

# 处理网格、基函数相关
include("MeshAndBFs.jl")

## 源信息
include("Sources/Source.jl")
include("Sources/Port.jl")

## 计算质量检查
include("SanityChecks.jl")

## Green's Function module (layered media and ground plane support)
include("GreenFunction/GreenFunction.jl")
@reexport using .GreenFunction

end
