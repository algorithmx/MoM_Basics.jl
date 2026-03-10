
@inline function _normalize_port_vector(v::MVec3D{FT}) where {FT<:Real}
    vn = norm(v)
    vn > zero(FT) || error("Port direction vectors must be non-zero")
    return v ./ vn
end

function _default_width_direction(normal::MVec3D{FT}) where {FT<:Real}
    ref = abs(normal[1]) < FT(0.9) ? MVec3D{FT}(1, 0, 0) : MVec3D{FT}(0, 1, 0)
    width_dir = ref - (ref ⋅ normal) * normal
    return _normalize_port_vector(width_dir)
end

function _rectangular_port_axes(
    normal::MVec3D{FT},
    widthDir::MVec3D{FT}
) where {FT<:Real}
    n̂ = _normalize_port_vector(normal)
    wdir = widthDir
    if iszero(wdir)
        wdir = _default_width_direction(n̂)
    else
        wdir = wdir - (wdir ⋅ n̂) * n̂
        wdir = _normalize_port_vector(wdir)
    end
    hdir = _normalize_port_vector(cross(n̂, wdir))
    return n̂, wdir, hdir
end

@inline function _rectangular_port_local_coordinates(
    point::MVec3D{FT},
    center::MVec3D{FT},
    widthDir::MVec3D{FT},
    heightDir::MVec3D{FT},
    normal::MVec3D{FT}
) where {FT<:Real}
    rel = point - center
    return rel ⋅ widthDir, rel ⋅ heightDir, rel ⋅ normal
end

@inline function _point_in_rectangular_port_box(
    point::MVec3D{FT},
    center::MVec3D{FT},
    widthDir::MVec3D{FT},
    heightDir::MVec3D{FT},
    normal::MVec3D{FT},
    width::FT,
    height::FT,
    tol::FT
) where {FT<:Real}
    u, v, w = _rectangular_port_local_coordinates(point, center, widthDir, heightDir, normal)
    return abs(u) <= width / 2 + tol &&
           abs(v) <= height / 2 + tol &&
           abs(w) <= tol
end

function _collect_rectangular_port_vertices(
    center::MVec3D{FT},
    widthDir::MVec3D{FT},
    heightDir::MVec3D{FT},
    normal::MVec3D{FT},
    width::FT,
    height::FT,
    tol::FT,
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    vertex_ids = Set{IT}()
    for tri in trianglesInfo
        for local_vid in 1:3
            vertex = MVec3D{FT}(tri.vertices[:, local_vid])
            if _point_in_rectangular_port_box(vertex, center, widthDir, heightDir, normal, width, height, tol)
                push!(vertex_ids, tri.verticesID[local_vid])
            end
        end
    end
    return sort!(collect(vertex_ids))
end

function _collect_rectangular_port_triangles(
    vertex_ids::Vector{IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    vertex_set = Set(vertex_ids)
    tri_ids = IT[]
    for tri in trianglesInfo
        if all(vid -> vid in vertex_set, tri.verticesID)
            push!(tri_ids, tri.triID)
        end
    end
    return tri_ids
end

function _rwg_edge_geometry(
    rwg::RWG{IT, FT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    tri_slot = rwg.inGeo[1] != 0 ? 1 : 2
    tri_id = rwg.inGeo[tri_slot]
    tri_id != 0 || error("RWG $(rwg.bfID) has no attached triangle")
    local_edge_id = rwg.inGeoID[tri_slot]
    tri = trianglesInfo[tri_id]

    vminus = EDGEVmINTriVsID[local_edge_id]
    vplus = EDGEVpINTriVsID[local_edge_id]
    vid1 = tri.verticesID[vminus]
    vid2 = tri.verticesID[vplus]

    p1 = MVec3D{FT}(tri.vertices[:, vminus])
    p2 = MVec3D{FT}(tri.vertices[:, vplus])
    center = MVec3D{FT}((p1 + p2) / 2)
    orient = _normalize_port_vector(MVec3D{FT}(p2 - p1))

    return vid1, vid2, center, orient
end

@inline function _rectangular_te10_weight(
    edge_center::MVec3D{FT},
    port_center::MVec3D{FT},
    widthDir::MVec3D{FT},
    heightDir::MVec3D{FT},
    normal::MVec3D{FT},
    width::FT
) where {FT<:Real}
    u, _, _ = _rectangular_port_local_coordinates(edge_center, port_center, widthDir, heightDir, normal)
    ξ = u / width + FT(0.5)
    return Complex{FT}(sin(pi * ξ))
end

function _collect_rectangular_port_edges(
    vertex_ids::Vector{IT},
    triangle_ids::Vector{IT},
    center::MVec3D{FT},
    widthDir::MVec3D{FT},
    heightDir::MVec3D{FT},
    normal::MVec3D{FT},
    width::FT,
    rwgsInfo::Vector{RWG{IT, FT}},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}
    vertex_set = Set(vertex_ids)
    triangle_set = Set(triangle_ids)

    rwg_ids = IT[]
    tri_id_pos = IT[]
    tri_id_neg = IT[]
    edge_lengths = FT[]
    edge_centers = MVec3D{FT}[]
    edge_orients = MVec3D{FT}[]
    edge_weights = Complex{FT}[]

    for rwg in rwgsInfo
        tri_pos_selected = rwg.inGeo[1] != 0 && rwg.inGeo[1] in triangle_set
        tri_neg_selected = rwg.inGeo[2] != 0 && rwg.inGeo[2] in triangle_set
        (tri_pos_selected ⊻ tri_neg_selected) || continue

        vid1, vid2, edge_center, edge_orient = _rwg_edge_geometry(rwg, trianglesInfo)
        (vid1 in vertex_set && vid2 in vertex_set) || continue

        push!(rwg_ids, rwg.bfID)
        push!(tri_id_pos, rwg.inGeo[1])
        push!(tri_id_neg, rwg.inGeo[2])
        push!(edge_lengths, rwg.edgel)
        push!(edge_centers, edge_center)
        push!(edge_orients, edge_orient)
        push!(edge_weights, _rectangular_te10_weight(edge_center, center, widthDir, heightDir, normal, width))
    end

    return rwg_ids, tri_id_pos, tri_id_neg, edge_lengths, edge_centers, edge_orients, edge_weights
end


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
