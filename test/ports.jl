# 端口功能测试模块
# 本测试文件测试以下功能：
# 1. DeltaGapPort 类型的创建和基本操作
# 2. CurrentProbe 类型的创建和基本操作
# 3. PortArray 端口数组的管理功能
# 4. 激励向量计算功能
#
# NOTE: S-parameter tests (computeInputImpedance, computeS11, computeSParameters)
# have been moved to MoM_Kernels.jl/test/ since they are now defined in MoM_Kernels.

using StaticArrays

function _build_planar_grid(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}; keep_half_rwg::Bool = false)
    nx = length(xs)
    ny = length(ys)
    nodes = zeros(Float64, 3, nx * ny)

    for j in 1:ny
        for i in 1:nx
            idx = (j - 1) * nx + i
            nodes[:, idx] = [Float64(xs[i]), Float64(ys[j]), 0.0]
        end
    end

    triangles = Matrix{Int}(undef, 3, 2 * (nx - 1) * (ny - 1))
    tri_idx = 1
    for j in 1:(ny - 1)
        for i in 1:(nx - 1)
            n1 = (j - 1) * nx + i
            n2 = n1 + 1
            n3 = j * nx + i
            n4 = n3 + 1

            triangles[:, tri_idx] = [n1, n2, n4]
            tri_idx += 1
            triangles[:, tri_idx] = [n1, n4, n3]
            tri_idx += 1
        end
    end

    mesh = TriangleMesh(size(triangles, 2), nodes, triangles)
    return MoM_Basics.getTriangleInfo(mesh; keep_half_rwg = keep_half_rwg)
end

@testset "DeltaGapPort" begin

    # 设置精度
    setPrecision!(Float64)

    # 创建一个简单的三角形网格用于测试
    # 三角形顶点坐标
    vertices = Float64[0 1 0.5;
                        0 0 1;
                        0 0 0]
    triangles = [1 2 3]

    # 创建 TriangleMesh
    mesh = TriangleMesh(1, vertices, triangles)

    # 创建 TriangleInfo
    triInfo = TriangleInfo{Int, Float64}(1)
    triInfo.vertices = vertices
    triInfo.center = [0.5, 0.333333, 0.0]
    triInfo.area = 0.5

    # 计算三角形参数
    setTriParam!(triInfo)

    # 创建 RWG 基函数
    rwg = RWG{Int, Float64}()
    rwg.bfID = 1
    rwg.edgel = 1.0
    rwg.inGeo = MVector{2, Int}(1, 0)  # 只有一个三角形
    rwg.center = MVector{3, Float64}(0.5, 0.333333, 0.0)

    # 测试基本构造函数
    port1 = DeltaGapPort{Float64, Int}(
        id = 1,
        V = ComplexF64(1.0, 0.0),
        freq = 2.4e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 0,
        edgel = 1.0,
        position = MVector{3, Float64}(0.5, 0.333333, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    @test port1.id == 1
    @test port1.V == 1.0 + 0.0im
    @test port1.freq == 2.4e9
    @test port1.rwgID == 1
    @test port1.isActive == true
    @test port1.portType == :delta_gap

    # 测试默认构造函数
    port2 = DeltaGapPort(
        id = 2,
        V = ComplexF64(2.0),
        freq = 3.0e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 0,
        edgel = 1.0,
        position = MVector{3, Float64}(0.5, 0.333333, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0)
    )

    @test port2.id == 2
    @test port2.V == 2.0 + 0.0im

    # 测试 find_edge_index 函数
    trianglesInfo = [triInfo]
    rwgsInfo = [rwg]

    result = find_edge_index(
        MVector{3, Float64}(0.5, 0.333333, 0.0),
        MVector{3, Float64}(1.0, 0.0, 0.0),
        trianglesInfo,
        rwgsInfo
    )

    @test result.rwgID == 1

    # 测试激励向量计算 (半基函数: triID_neg = 0, 所以激励 = V * edgel)
    nbf = 10
    V = excitationVectorEFIE(port1, trianglesInfo, nbf)

    @test length(V) == nbf
    @test V[1] ≈ 1.0  # 半基函数: V * l = 1.0 * 1.0 = 1.0

    # 测试激励向量 in-place 计算
    V2 = zeros(ComplexF64, nbf)
    excitationVectorEFIE!(V2, port1, trianglesInfo)

    @test V2[1] ≈ 1.0
    @test V2[2] ≈ 0.0

    # 测试 getPortVoltage
    voltage = getPortVoltage(port1, ComplexF64(0.01))
    @test voltage == 1.0 + 0.0im

    # 测试 getPortCurrent
    current = getPortCurrent(port1; Z = ComplexF64(50.0))
    @test current ≈ 0.02 + 0.0im

    # 测试 sourceEfield 和 sourceHfield
    e_field = sourceEfield(port1, [0.0, 0.0, 0.0])
    @test norm(e_field) ≈ 0.0

    h_field = sourceHfield(port1, [0.0, 0.0, 0.0])
    @test norm(h_field) ≈ 0.0

end


@testset "CurrentProbe" begin

    setPrecision!(Float64)

    # 创建电流探针
    probe1 = CurrentProbe{Float64, Int}(
        id = 1,
        I = ComplexF64(1.0, 0.0),
        freq = 2.4e9,
        rwgID = 5,
        triID = 3,
        edgel = 0.5,
        center = MVector{3, Float64}(0.25, 0.25, 0.0),
        isActive = true
    )

    @test probe1.id == 1
    @test probe1.I == 1.0 + 0.0im
    @test probe1.rwgID == 5
    @test probe1.isActive == true

    # 测试默认构造函数
    probe2 = CurrentProbe(
        id = 2,
        I = ComplexF64(0.5),
        freq = 1.0e9,
        rwgID = 10
    )

    @test probe2.id == 2
    @test probe2.I == 0.5 + 0.0im

    # 测试激励向量计算
    trianglesInfo = TriangleInfo{Int, Float64}[]
    nbf = 20
    V = excitationVectorEFIE(probe1, trianglesInfo, nbf)

    @test length(V) == nbf
    @test V[5] ≈ 1.0 + 0.0im  # 电流探针直接设置对应基函数的激励

    # 测试多个基函数位置为0
    @test V[1] ≈ 0.0
    @test V[10] ≈ 0.0

    # 测试 in-place 计算
    V2 = zeros(ComplexF64, nbf)
    excitationVectorEFIE!(V2, probe1, trianglesInfo)

    @test V2[5] ≈ 1.0

end


@testset "PortArray" begin

    setPrecision!(Float64)

    # 创建测试端口
    port1 = DeltaGapPort{Float64, Int}(
        id = 1,
        V = ComplexF64(1.0),
        freq = 2.4e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 0,
        edgel = 1.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    port2 = DeltaGapPort{Float64, Int}(
        id = 2,
        V = ComplexF64(2.0),
        freq = 2.4e9,
        rwgID = 2,
        triID_pos = 2,
        triID_neg = 0,
        edgel = 1.0,
        position = MVector{3, Float64}(1.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    probe1 = CurrentProbe{Float64, Int}(
        id = 3,
        I = ComplexF64(0.5),
        freq = 2.4e9,
        rwgID = 3,
        triID = 1,
        edgel = 0.5,
        center = MVector{3, Float64}(0.5, 0.0, 0.0),
        isActive = false  # 被动端口
    )

    # 创建端口数组
    ports = [port1, port2, probe1]
    portArray = PortArray(ports)

    @test portArray.numPorts == 3
    @test length(portArray.activePorts) == 2  # port1, port2
    @test length(portArray.passivePorts) == 1  # probe1

    # 测试 getExcitationVector (半基函数: triID_neg = 0)
    trianglesInfo = TriangleInfo{Int, Float64}[]
    nbf = 10
    V_total = getExcitationVector(portArray, trianglesInfo, nbf)

    @test length(V_total) == nbf
    @test V_total[1] ≈ 1.0  # port1: 1.0 * 1.0 = 1.0 (半基函数)
    @test V_total[2] ≈ 2.0  # port2: 2.0 * 1.0 = 2.0 (半基函数)
    @test V_total[3] ≈ 0.5  # probe1: 0.5

    # 测试 addExcitationVector!
    V_add = zeros(ComplexF64, nbf)
    addExcitationVector!(V_add, portArray, trianglesInfo)

    @test V_add[1] ≈ 1.0
    @test V_add[2] ≈ 2.0
    @test V_add[3] ≈ 0.5

end


@testset "Port Excitation Physics" begin

    setPrecision!(Float64)

    # 物理验证测试：Delta-Gap 端口激励
    # ============================================
    # 第一性原理验证：
    # 完整RWG基函数: V_m = V_0 × l_m / 2
    # 半基函数: V_m = V_0 × l_m
    # 电流探针: V_m = I_0

    # 测试1: 完整RWG基函数 (triID_neg > 0)
    # --------------------------------------------------------
    # 边长 l = 2.0m, 电压 V = 1.0V
    # 预期: V_m = 1.0 × 2.0 / 2 = 1.0

    rwg = RWG{Int, Float64}()
    rwg.bfID = 1
    rwg.edgel = 2.0
    rwg.inGeo = MVector{2, Int}(1, 2)  # 两个三角形 - 完整RWG
    rwg.center = MVector{3, Float64}(0.0, 0.0, 0.0)

    port = DeltaGapPort{Float64, Int}(
        id = 1,
        V = ComplexF64(1.0),
        freq = 1.0e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 2,  # 完整RWG基函数
        edgel = 2.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    trianglesInfo = TriangleInfo{Int, Float64}[]
    nbf = 5
    V = excitationVectorEFIE(port, trianglesInfo, nbf)

    # 验证公式: V_m = V_0 × l_m / 2
    expected = 1.0 * 2.0 / 2  # = 1.0
    @test V[1] ≈ expected

    # 测试2: 不同参数的完整RWG基函数
    # --------------------------------------------------------
    # 边长 l = 3.0m, 电压 V = 2.0V
    # 预期: V_m = 2.0 × 3.0 / 2 = 3.0

    port2 = DeltaGapPort{Float64, Int}(
        id = 2,
        V = ComplexF64(2.0),
        freq = 1.0e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 2,
        edgel = 3.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    V2 = excitationVectorEFIE(port2, trianglesInfo, nbf)
    expected2 = 2.0 * 3.0 / 2  # = 3.0
    @test V2[1] ≈ expected2

    # 测试3: 半基函数情况 (triID_neg = 0)
    # --------------------------------------------------------
    # 边长 l = 2.0m, 电压 V = 1.0V
    # 预期: V_m = V_0 × l_m = 1.0 × 2.0 = 2.0

    port_half = DeltaGapPort{Float64, Int}(
        id = 3,
        V = ComplexF64(1.0),
        freq = 1.0e9,
        rwgID = 2,
        triID_pos = 1,
        triID_neg = 0,  # 半基函数 - 边界元
        edgel = 2.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    V_half = excitationVectorEFIE(port_half, trianglesInfo, nbf)

    # 验证公式: V_m = V_0 × l_m
    expected_half = 1.0 * 2.0
    @test V_half[2] ≈ expected_half

    # 测试4: 不同参数的半基函数
    # --------------------------------------------------------
    # 边长 l = 0.5m, 电压 V = 5.0V
    # 预期: V_m = 5.0 × 0.5 = 2.5

    port_half2 = DeltaGapPort{Float64, Int}(
        id = 4,
        V = ComplexF64(5.0),
        freq = 1.0e9,
        rwgID = 3,
        triID_pos = 1,
        triID_neg = 0,  # 半基函数
        edgel = 0.5,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    V_half2 = excitationVectorEFIE(port_half2, trianglesInfo, nbf)
    expected_half2 = 5.0 * 0.5  # = 2.5
    @test V_half2[3] ≈ expected_half2

    # 测试5: 电流探针物理
    # --------------------------------------------------------
    # 电流 I = 1.0A
    # 预期: V_m = I_0 = 1.0

    probe = CurrentProbe{Float64, Int}(
        id = 1,
        I = ComplexF64(1.0),
        freq = 1.0e9,
        rwgID = 3,
        triID = 1,
        edgel = 1.0,
        center = MVector{3, Float64}(0.0, 0.0, 0.0),
        isActive = true
    )

    V_probe = excitationVectorEFIE(probe, trianglesInfo, nbf)

    # 验证公式: V_m = I_0
    @test V_probe[3] ≈ 1.0

    # 测试6: 不同参数的电流探针
    # --------------------------------------------------------
    # 电流 I = 2.5A
    # 预期: V_m = I_0 = 2.5

    probe2 = CurrentProbe{Float64, Int}(
        id = 2,
        I = ComplexF64(2.5),
        freq = 1.0e9,
        rwgID = 4,
        triID = 1,
        edgel = 1.0,
        center = MVector{3, Float64}(0.0, 0.0, 0.0),
        isActive = true
    )

    V_probe2 = excitationVectorEFIE(probe2, trianglesInfo, nbf)
    @test V_probe2[4] ≈ 2.5

    # 测试7: 复数电压/电流激励
    # --------------------------------------------------------

    # 复数电压: V = 1.0 + 1.0im
    port_complex = DeltaGapPort{Float64, Int}(
        id = 5,
        V = ComplexF64(1.0, 1.0),
        freq = 1.0e9,
        rwgID = 5,
        triID_pos = 1,
        triID_neg = 2,
        edgel = 2.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    V_complex = excitationVectorEFIE(port_complex, trianglesInfo, nbf)
    expected_complex = (1.0 + 1.0im) * 2.0 / 2  # = 1.0 + 1.0im
    @test V_complex[5] ≈ expected_complex

    # 测试8: 复数电流: I = 1.0 + 2.0im
    probe_complex = CurrentProbe{Float64, Int}(
        id = 3,
        I = ComplexF64(1.0, 2.0),
        freq = 1.0e9,
        rwgID = 5,
        triID = 1,
        edgel = 1.0,
        center = MVector{3, Float64}(0.0, 0.0, 0.0),
        isActive = true
    )

    V_probe_complex = excitationVectorEFIE(probe_complex, trianglesInfo, nbf)
    @test V_probe_complex[5] ≈ ComplexF64(1.0, 2.0)

end


@testset "RectangularEdgePort detection" begin

    setPrecision!(Float64)

    trianglesInfo, rwgsInfo = _build_planar_grid(0.0:1.0:4.0, 0.0:1.0:3.0)
    port = RectangularEdgePort{Float64, Int}(
        id = 1,
        V = ComplexF64(1.0),
        freq = 1.0e9,
        center = MVector{3, Float64}(2.0, 1.5, 0.0),
        normal = MVector{3, Float64}(0.0, 0.0, 1.0),
        width = 2.0,
        height = 1.0,
        widthDirection = MVector{3, Float64}(1.0, 0.0, 0.0),
        tol = 1e-8,
        trianglesInfo = trianglesInfo,
        rwgsInfo = rwgsInfo
    )

    @test port.portType == :rectangular_edge
    @test port.mode == :TE10
    @test port.vertexIDs == [7, 8, 9, 12, 13, 14]
    @test length(port.triangleIDs) == 4
    @test length(port.rwgIDs) == 6

    sorted_weights = sort(abs.(port.edgeWeights))
    expected_weights = [0.0, 0.0, sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5)]
    @test all(isapprox.(sorted_weights, expected_weights; atol = 1e-10))
end

@testset "RectangularEdgePort boundary excitation" begin

    setPrecision!(Float64)

    trianglesInfo, rwgsInfo = _build_planar_grid([0.0, 1.0], [0.0, 1.0]; keep_half_rwg = true)
    @test any(rwg -> rwg.isbd, rwgsInfo)

    port = RectangularEdgePort{Float64, Int}(
        id = 2,
        V = ComplexF64(1.0),
        freq = 1.0e9,
        center = MVector{3, Float64}(0.5, 0.5, 0.0),
        normal = MVector{3, Float64}(0.0, 0.0, 1.0),
        width = 1.0,
        height = 1.0,
        widthDirection = MVector{3, Float64}(1.0, 0.0, 0.0),
        tol = 1e-8,
        trianglesInfo = trianglesInfo,
        rwgsInfo = rwgsInfo
    )

    @test length(port.rwgIDs) == 4
    @test all(==(0), port.triID_neg)
    @test all(isapprox.(sort(abs.(port.edgeWeights)), [0.0, 0.0, 1.0, 1.0]; atol = 1e-10))

    V_rect = excitationVectorEFIE(port, trianglesInfo, length(rwgsInfo))
    @test all(isapprox.(sort(abs.(V_rect[port.rwgIDs])), [0.0, 0.0, 1.0, 1.0]; atol = 1e-10))
end

@testset "RectangularEdgePort PortArray integration" begin

    setPrecision!(Float64)

    trianglesInfo, rwgsInfo = _build_planar_grid([0.0, 1.0], [0.0, 1.0]; keep_half_rwg = true)
    rect_port = RectangularEdgePort{Float64, Int}(
        id = 3,
        V = ComplexF64(2.0),
        freq = 1.0e9,
        center = MVector{3, Float64}(0.5, 0.5, 0.0),
        normal = MVector{3, Float64}(0.0, 0.0, 1.0),
        width = 1.0,
        height = 1.0,
        widthDirection = MVector{3, Float64}(1.0, 0.0, 0.0),
        tol = 1e-8,
        trianglesInfo = trianglesInfo,
        rwgsInfo = rwgsInfo
    )

    spare_rwg = first(setdiff(collect(1:length(rwgsInfo)), rect_port.rwgIDs))
    probe = CurrentProbe{Float64, Int}(
        id = 4,
        I = ComplexF64(0.5),
        freq = 1.0e9,
        rwgID = spare_rwg,
        triID = rwgsInfo[spare_rwg].inGeo[1],
        edgel = rwgsInfo[spare_rwg].edgel,
        center = rwgsInfo[spare_rwg].center,
        isActive = false
    )

    ports = PortType[rect_port, probe]
    portArray = PortArray(ports)
    V_total = getExcitationVector(portArray, trianglesInfo, length(rwgsInfo))
    V_expected = excitationVectorEFIE(rect_port, trianglesInfo, length(rwgsInfo)) .+
                 excitationVectorEFIE(probe, trianglesInfo, length(rwgsInfo))

    @test portArray.numPorts == 2
    @test length(portArray.activePorts) == 1
    @test length(portArray.passivePorts) == 1
    @test V_total ≈ V_expected
end

@testset "S-Parameter Calculation" begin

    setPrecision!(Float64)

    # ============================================================
    # S-参数计算测试
    # 物理原理：
    # 1. 输入阻抗: Z_in = V_port / I_port (从 MoM 阻抗矩阵求解)
    # 2. S11: S11 = (Z_in - Z0) / (Z_in + Z0)
    # 3. 多端口: S = (Z - Z0*I) * (Z + Z0*I)^(-1)
    # ============================================================

    # 测试1: 理想50欧姆匹配负载
    # --------------------------------------------------------
    # 当 Z_in = Z0 = 50Ω 时，S11 = 0 (完全匹配，无反射)
    nbf = 3

    # 构造一个简单的阻抗矩阵
    Z_matrix = ComplexF64[
        50+0im  0    0
        0     50+0im  0
        0     0     50+0im
    ]

    # 创建端口
    port1 = DeltaGapPort{Float64, Int}(
        id = 1,
        V = ComplexF64(1.0),
        freq = 1.0e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 2,
        edgel = 1.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )

    # 激励向量
    V_excitation = zeros(ComplexF64, nbf)
    V_excitation[1] = 1.0

    # NOTE: S-parameter tests (computeInputImpedance, computeS11, computeSParameters)
    # are located in MoM_Kernels.jl/test/sparameters.jl

    # ============================================================
    # MFIE/CFIE 端口激励使用 EFIE (2026-02-28 修复)
    # ============================================================
    # 验证 MFIE 和 CFIE 公式现在使用 EFIE 激励向量
    # 而不是硬编码的 50Ω 假设
    
    using MoM_Basics: EFIE, MFIE, CFIE, assembleExcitationVector!
    
    # 创建测试端口数组
    test_port_formulation = DeltaGapPort{Float64, Int}(
        id = 99,
        V = ComplexF64(1.0),
        freq = 1.0e9,
        rwgID = 1,
        triID_pos = 1,
        triID_neg = 2,  # 全 RWG 基函数
        edgel = 1.0,
        position = MVector{3, Float64}(0.0, 0.0, 0.0),
        direction = MVector{3, Float64}(1.0, 0.0, 0.0),
        isActive = true
    )
    
    test_ports_formulation = PortArray([test_port_formulation])
    tri_info_formulation = TriangleInfo{Int, Float64}[]
    nbf_formulation = 5
    
    # 测试主要目标：验证 MFIE/CFIE 使用电压端口时会发出警告
    
    # MFIE 激励向量 - 应该发出警告
    V_mfie_test = zeros(ComplexF64, nbf_formulation)
    @test_logs (:warn, r"MFIE formulation detected.*EFIE excitation") begin
        assembleExcitationVector!(V_mfie_test, test_ports_formulation, tri_info_formulation, nbf_formulation, MFIE())
    end
    
    # CFIE 激励向量 - 应该发出警告
    V_cfie_test = zeros(ComplexF64, nbf_formulation)
    @test_logs (:warn, r"CFIE formulation detected.*EFIE excitation") begin
        assembleExcitationVector!(V_cfie_test, test_ports_formulation, tri_info_formulation, nbf_formulation, CFIE{Float64}(alpha=0.5))
    end
    
    # 验证函数不报错，返回的激励向量是正确格式
    @test length(V_mfie_test) == nbf_formulation
    @test length(V_cfie_test) == nbf_formulation
    @test eltype(V_mfie_test) == ComplexF64
    @test eltype(V_cfie_test) == ComplexF64

end
