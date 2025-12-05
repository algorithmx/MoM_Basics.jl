"""
    ExcitationFieldData{FT, CT}

Structure to store excitation field data at specific points.
"""
struct ExcitationFieldData{FT<:AbstractFloat, CT<:Complex{FT}}
    npoints     ::Int
    positions   ::Vector{SVec3D{FT}}
    E           ::Vector{SVec3D{CT}}
    H           ::Vector{SVec3D{CT}}
end

"""
    calExcitationFields(geosInfo, source::ExcitingSource)

Calculate E and H fields from `source` at the centroids of geometry elements in `geosInfo`.
Returns an `ExcitationFieldData` object.
"""
function calExcitationFields(geosInfo, source::ExcitingSource)
    # Flatten nested vectors if necessary
    geos_flat = []
    if eltype(geosInfo) <: TriangleInfo
        geos_flat = geosInfo
    elseif eltype(geosInfo) <: Vector
        for part in geosInfo
            append!(geos_flat, part)
        end
    else
         # Try to iterate assuming it's iterable
         for geo in geosInfo
             push!(geos_flat, geo)
         end
    end
    
    npoints = length(geos_flat)
    
    # Determine types from the first element or use Precision.FT
    FT = Precision.FT
    CT = Complex{FT}
    
    positions = Vector{SVec3D{FT}}(undef, npoints)
    E         = Vector{SVec3D{CT}}(undef, npoints)
    H         = Vector{SVec3D{CT}}(undef, npoints)

    Threads.@threads for i in 1:npoints
        geo = geos_flat[i]
        if hasproperty(geo, :center)
            r = SVec3D{FT}(geo.center) # Convert MVec to SVec if needed
            positions[i] = r
            E[i] = sourceEfield(source, r)
            H[i] = sourceHfield(source, r)
        end
    end
    
    return ExcitationFieldData{FT, CT}(npoints, positions, E, H)
end

"""
    saveExcitationFields(filename::String, data::ExcitationFieldData)

Save excitation fields to a file. Supports CSV (.csv) and NPZ (.npz) formats.
"""
function saveExcitationFields(filename::String, data::ExcitationFieldData)
    if endswith(filename, ".npz")
        n = data.npoints
        
        # Preallocate matrices (N x 3)
        # data.positions elements are SVec3D{FT} (which index as [1], [2], [3])
        pos_arr = Matrix{eltype(eltype(data.positions))}(undef, n, 3)
        E_arr   = Matrix{eltype(eltype(data.E))}(undef, n, 3)
        H_arr   = Matrix{eltype(eltype(data.H))}(undef, n, 3)
        
        for i in 1:n
            pos_arr[i, 1] = data.positions[i][1]
            pos_arr[i, 2] = data.positions[i][2]
            pos_arr[i, 3] = data.positions[i][3]
            
            E_arr[i, 1] = data.E[i][1]
            E_arr[i, 2] = data.E[i][2]
            E_arr[i, 3] = data.E[i][3]
            
            H_arr[i, 1] = data.H[i][1]
            H_arr[i, 2] = data.H[i][2]
            H_arr[i, 3] = data.H[i][3]
        end
        
        npzwrite(filename, Dict("positions" => pos_arr, "E" => E_arr, "H" => H_arr))
        
    else
        open(filename, "w") do io
            println(io, "rx,ry,rz,Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag,Hx_real,Hx_imag,Hy_real,Hy_imag,Hz_real,Hz_imag")
            for i in 1:data.npoints
                r = data.positions[i]
                e = data.E[i]
                h = data.H[i]
                @printf io "%.6e,%.6e,%.6e," r[1] r[2] r[3]
                @printf io "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e," real(e[1]) imag(e[1]) real(e[2]) imag(e[2]) real(e[3]) imag(e[3])
                @printf io "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n" real(h[1]) imag(h[1]) real(h[2]) imag(h[2]) real(h[3]) imag(h[3])
            end
        end
    end
    nothing
end

"""
    saveExcitationFields(filename::String, geosInfo, source::ExcitingSource)

Calculate and save excitation fields to a CSV file.
"""
function saveExcitationFields(filename::String, geosInfo, source::ExcitingSource)
    data = calExcitationFields(geosInfo, source)
    saveExcitationFields(filename, data)
end
