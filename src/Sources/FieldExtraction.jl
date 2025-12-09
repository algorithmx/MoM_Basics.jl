"""
    FieldData{FT, CT}

Unified structure to store field data (incident fields, currents, etc.) at specific points.
"""
struct FieldData{FT<:AbstractFloat, CT<:Complex{FT}}
    npoints     ::Int
    positions   ::Vector{SVec3D{FT}}
    fields      ::Dict{Symbol, Vector{SVec3D{CT}}}
end

FieldData{FT, CT}(npoints::Int, positions::Vector{SVec3D{FT}}) where {FT, CT} = 
    FieldData{FT, CT}(npoints, positions, Dict{Symbol, Vector{SVec3D{CT}}}())

"""
    calIncidentFields(geosInfo, source::ExcitingSource)

Calculate E and H incident fields from `source` at the centroids of geometry elements.
Returns `FieldData`.
"""
function calIncidentFields(geosInfo, source::ExcitingSource)
    # Flatten geometry
    geos_flat = _flatten_geos_basics(geosInfo)
    
    npoints = length(geos_flat)
    FT = Precision.FT
    CT = Complex{FT}
    
    positions = Vector{SVec3D{FT}}(undef, npoints)
    E         = Vector{SVec3D{CT}}(undef, npoints)
    H         = Vector{SVec3D{CT}}(undef, npoints)

    Threads.@threads for i in 1:npoints
        geo = geos_flat[i]
        if hasproperty(geo, :center)
            r = SVec3D{FT}(geo.center)
            positions[i] = r
            E[i] = sourceEfield(source, r)
            H[i] = sourceHfield(source, r)
        end
    end
    
    fd = FieldData{FT, CT}(npoints, positions)
    fd.fields[:E_inc] = E
    fd.fields[:H_inc] = H
    return fd
end

function _flatten_geos_basics(geosInfo::AbstractVector{<:VSCellType})
    return geosInfo
end

function _flatten_geos_basics(geosInfo::AbstractVector{<:AbstractVector})
    return reduce(vcat, geosInfo)
end

function _flatten_geos_basics(geosInfo)
    # Fallback for generic iterables
    geos_flat = []
    for part in geosInfo
        if isa(part, AbstractVector)
             append!(geos_flat, part)
        else
             push!(geos_flat, part)
        end
    end
    return geos_flat
end

# Backward compatibility aliases
calExcitationFields(geosInfo, source) = calIncidentFields(geosInfo, source)


# Internal: evaluate RWG basis value at r on a given triangle with local-id idx_in_geo (1..3)
@inline function _rwg_value_at(r::SVec3D{FT}, bf, tri, idx_in_geo::Int) where {FT}
    sgn = sign(tri.edgel[idx_in_geo])
    l   = bf.edgel
    A   = tri.area
    r_free = SVec3D{FT}(tri.vertices[:, idx_in_geo])
    return (sgn * l / (2A)) * (r - r_free)
end


"""
    mergeFieldData!(target::FieldData, source::FieldData)

Merge fields from `source` into `target`. Requires matching number of points.
Note: Does not rigorously check if positions are identical, assumes consistent mesh usage.
"""
function mergeFieldData!(target::FieldData, source::FieldData)
    if target.npoints != source.npoints
        error("Cannot merge FieldData: different number of points (target: $(target.npoints), source: $(source.npoints))")
    end
    merge!(target.fields, source.fields)
    return target
end

"""
    saveFieldData(filename::String, data::FieldData)

Save field data to CSV or NPZ.
"""
function saveFieldData(filename::String, data::FieldData)
    if endswith(filename, ".npz")
        n = data.npoints
        dict_to_save = Dict{String, Any}()
        
        # Save Positions
        pos_arr = Matrix{eltype(eltype(data.positions))}(undef, n, 3)
        for i in 1:n
            pos_arr[i, 1] = data.positions[i][1]
            pos_arr[i, 2] = data.positions[i][2]
            pos_arr[i, 3] = data.positions[i][3]
        end
        dict_to_save["positions"] = pos_arr
        
        # Save Fields
        for (key, val) in data.fields
             f_arr = Matrix{eltype(eltype(val))}(undef, n, 3)
             for i in 1:n
                 f_arr[i, 1] = val[i][1]
                 f_arr[i, 2] = val[i][2]
                 f_arr[i, 3] = val[i][3]
             end
             dict_to_save[string(key)] = f_arr
        end
        
        npzwrite(filename, dict_to_save)
        
    else
        open(filename, "w") do io
            # Construct Header
            header = "rx,ry,rz"
            sorted_keys = sort(collect(keys(data.fields)))
            for k in sorted_keys
                k_str = string(k)
                header *= ",$(k_str)x_real,$(k_str)x_imag,$(k_str)y_real,$(k_str)y_imag,$(k_str)z_real,$(k_str)z_imag"
            end
            println(io, header)
            
            for i in 1:data.npoints
                r = data.positions[i]
                @printf io "%.6e,%.6e,%.6e" r[1] r[2] r[3]
                
                for k in sorted_keys
                    val = data.fields[k][i]
                    @printf io ",%.6e,%.6e,%.6e,%.6e,%.6e,%.6e" real(val[1]) imag(val[1]) real(val[2]) imag(val[2]) real(val[3]) imag(val[3])
                end
                print(io, "\n")
            end
        end
    end
    nothing
end

# Wrappers for easier usage
function saveIncidentFields(filename::String, geosInfo, source::ExcitingSource)
    data = calIncidentFields(geosInfo, source)
    saveFieldData(filename, data)
end

# Backward compatibility wrappers (can be deprecated later)
saveExcitationFields(filename::String, geosInfo, source::ExcitingSource) = saveIncidentFields(filename, geosInfo, source)
saveExcitationFields(filename::String, data::FieldData) = saveFieldData(filename, data)
saveSurfaceCurrents(filename::String, data::FieldData) = saveFieldData(filename, data)
