# FreeSpaceGF.jl
# Free-space scalar Green's function implementation
# This is the existing behavior, refactored to use the unified interface

# =============================================================================
# Free-Space Green's Function Type
# =============================================================================

"""
    FreeSpaceGF{FT<:AbstractFloat} <: AbstractGreenFunction{FT}

Free-space scalar Green's function.

The Green's function in free space is:
```
G(R) = exp(-jkR) / R
```
where R = |r_obs - r_src| is the distance between observation and source points.

# Fields
- `k::Complex{FT}`: Wavenumber (complex to account for lossy media)

# Notes
- Both vector potential (g_A) and scalar potential (g_phi) are identical in free space
- This is the default Green's function used by the original MoM code
- Maintains backward compatibility with existing simulations
"""
struct FreeSpaceGF{FT<:AbstractFloat} <: AbstractGreenFunction{FT}
    k::Complex{FT}
end

# =============================================================================
# Core Evaluation Functions
# =============================================================================

"""
    evaluate_greenfunc(gf::FreeSpaceGF, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate free-space Green's function between two points.

# Mathematical Formulation
```
G(R) = exp(-jkR) / R
R = |r_obs - r_src|
```

# Returns
- `GreenFuncVals`: Both g_A and g_phi equal to G(R) (identical in free space)

# Notes
- Preserves existing greenfunc behavior from UsefulFunctions.jl
- Used as reference for validating other Green's function types
"""
function evaluate_greenfunc(gf::FreeSpaceGF{FT}, 
                             r_obs::AbstractVector, 
                             r_src::AbstractVector) where {FT}
    R = norm(r_obs - r_src)
    G = exp(-im * gf.k * R) / R
    return GreenFuncVals{Complex{FT}}(G)
end

function evaluate_greenfunc(gf::FreeSpaceGF{FT}, 
                             r_obs::Vec3D{FT}, 
                             r_src::Vec3D{FT}) where {FT}
    # Compute distance
    R = sqrt((r_obs[1]-r_src[1])^2 + (r_obs[2]-r_src[2])^2 + (r_obs[3]-r_src[3])^2)
    G = exp(-im * gf.k * R) / R
    return GreenFuncVals{Complex{FT}}(G)
end

# =============================================================================
# Distance-Based Evaluation (for efficiency)
# =============================================================================

"""
    evaluate_greenfunc(gf::FreeSpaceGF, R::Real)

Evaluate free-space Green's function given precomputed distance.

# Arguments
- `gf::FreeSpaceGF`: Green's function instance
- `R::Real`: Distance between points

# Returns
- `Complex`: G(R) = exp(-jkR) / R

# Notes
- More efficient when distance is already computed
- Used by existing code that precomputes distances
"""
function evaluate_greenfunc(gf::FreeSpaceGF{FT}, R::Real) where {FT}
    G = exp(-im * gf.k * R) / R
    return GreenFuncVals{Complex{FT}}(G)
end

# =============================================================================
# Singularity Handling
# =============================================================================

"""
    greenfunc_star_freespace(R::Real, k::Real; taylor_order::Int=15)

De-singularized Green's function for free space.

Computes: g*(R) = exp(-jkR)/R - 1/R = Σ_{n=1}^∞ (-jk)^n R^{n-1} / n!

# Arguments
- `R::Real`: Distance
- `k::Real`: Wavenumber
- `taylor_order::Int`: Order of Taylor expansion (default: 15)

# Returns
- `Complex`: Regular part of Green's function (1/R singularity removed)

# Notes
- Used for coincident triangle integrals where 1/R is extracted analytically
- Equivalent to greenfunc_star in Singularity.jl
- Maintains consistency with existing singularity handling
"""
function greenfunc_star_freespace(R::FT, k::Complex{FT}; taylor_order::Int=15) where {FT<:AbstractFloat}
    minusJk = -1im * k
    # Start with n=1 term: (-jk)^1 * R^0 / 1! = -jk
    g_star = minusJk
    temp0 = minusJk * R
    temp1 = minusJk
    
    # Taylor expansion to specified order
    if taylor_order >= 2
        @inbounds for i in 2:taylor_order
            temp1 *= temp0 / i
            g_star += temp1
        end
    end
    
    return g_star
end

"""
    evaluate_greenfunc_star(gf::FreeSpaceGF, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate de-singularized Green's function for free space.

# Returns
- `GreenFuncVals`: Both components with 1/R singularity removed

# Notes
- Delegates to greenfunc_star_freespace
- Returns same value for g_A and g_phi (free space)
"""
function evaluate_greenfunc_star(gf::FreeSpaceGF{FT}, 
                                  r_obs::AbstractVector, 
                                  r_src::AbstractVector;
                                  taylor_order::Int=15) where {FT}
    R = norm(r_obs - r_src)
    g_star = greenfunc_star_freespace(R, real(gf.k); taylor_order=taylor_order)
    return GreenFuncVals{Complex{FT}}(g_star)
end

function evaluate_greenfunc_star(gf::FreeSpaceGF{FT}, 
                                  r_obs::Vec3D{FT}, 
                                  r_src::Vec3D{FT};
                                  taylor_order::Int=15) where {FT}
    R = norm(r_obs - r_src)
    g_star = greenfunc_star_freespace(R, real(gf.k); taylor_order=taylor_order)
    return GreenFuncVals{Complex{FT}}(g_star)
end

# =============================================================================
# Legacy Compatibility
# =============================================================================

"""
    to_legacy_greenfunc(gf::FreeSpaceGF) -> Function

Return a function compatible with legacy greenfunc interface.

# Returns
- `Function`: (r_obs, r_src) -> exp(-jkR)/R

# Notes
- Allows gradual migration of existing code
- New code should use evaluate_greenfunc directly
"""
function to_legacy_greenfunc(gf::FreeSpaceGF)
    return (r_obs, r_src) -> begin
        vals = evaluate_greenfunc(gf, r_obs, r_src)
        return vals.g_A  # Same as g_phi in free space
    end
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    wavenumber(gf::FreeSpaceGF) -> Complex

Get the wavenumber used by this Green's function.
"""
wavenumber(gf::FreeSpaceGF) = gf.k

"""
    wavelength(gf::FreeSpaceGF) -> Real

Get the wavelength (2π/k).
"""
wavelength(gf::FreeSpaceGF{FT}) where {FT} = 2π / abs(real(gf.k))

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, gf::FreeSpaceGF{FT}) where {FT} = 
    print(io, "FreeSpaceGF{$FT}(k=$(gf.k))")

Base.summary(io::IO, gf::FreeSpaceGF{FT}) where {FT} = 
    print(io, "FreeSpaceGF{$FT} (λ=$(wavelength(gf)))")
