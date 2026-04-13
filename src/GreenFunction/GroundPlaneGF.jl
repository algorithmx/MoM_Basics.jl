# GroundPlaneGF.jl
# PEC Ground Plane Green's Function via Image Theory
# 
# Physical Model:
# - Infinite, flat, perfectly conducting ground plane at z = z_gnd
# - Image theory replaces ground with mirror sources below the plane
# - Horizontal currents: image has opposite sign (-1)
# - Vertical currents: image has same sign (+1) for vector potential
#
# Mathematical Formulation:
# For vector potential A (horizontal currents):
#   G_A = G_direct - G_image  (image current reversed)
#
# For scalar potential φ:
#   G_φ = G_direct - G_image  (image charge opposite sign)

# =============================================================================
# Ground Plane Green's Function Type
# =============================================================================

"""
    GroundPlaneGF{FT<:AbstractFloat} <: AbstractGreenFunction{FT}

Green's function for PEC ground plane using image theory.

# Fields
- `k::Complex{FT}`: Wavenumber
- `z_gnd::FT`: Ground plane z-coordinate

# Physical Interpretation
The ground plane is replaced by image sources reflected across z = z_gnd:
- Source at (x, y, z) has image at (x, y, 2*z_gnd - z)
- Image is below the ground plane (for sources above)

# Notes
- Both vector and scalar potentials use image contributions
# - For horizontal currents, both A and φ use a (-) sign
- Assumes all sources and observers are ABOVE the ground plane (z >= z_gnd)
- Image contributions are smooth (no singularity) when z > z_gnd

# References
- Balanis, "Advanced Engineering Electromagnetics", Sec. 11.4
- NEC-2 User's Guide, "Ground Parameters" (GN card)
"""
struct GroundPlaneGF{FT<:AbstractFloat} <: AbstractGreenFunction{FT}
    k::Complex{FT}
    z_gnd::FT
    
    function GroundPlaneGF{FT}(k::Complex{FT}, z_gnd::FT) where {FT<:AbstractFloat}
        # Validate that z_gnd is finite
        isfinite(z_gnd) || error("Ground plane z-coordinate must be finite")
        new(k, z_gnd)
    end
end

# Convenience constructor
GroundPlaneGF(k::Complex{FT}, z_gnd::FT) where {FT} = GroundPlaneGF{FT}(k, z_gnd)
GroundPlaneGF(k::Real, z_gnd::Real) = GroundPlaneGF(Complex(k), z_gnd)

# =============================================================================
# Core Evaluation Functions
# =============================================================================

"""
    evaluate_greenfunc(gf::GroundPlaneGF, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate Green's function with PEC ground plane using image theory.

# Mathematical Formulation
```
R_direct = |r_obs - r_src|
R_image  = |r_obs - r_src_image|

where r_src_image = mirror_point_across_ground(r_src, z_gnd)

G_direct = exp(-jk*R_direct) / R_direct
G_image  = exp(-jk*R_image) / R_image

G_A   = G_direct - G_image   (vector potential)
G_φ   = G_direct - G_image   (scalar potential)
```

# Arguments
- `gf::GroundPlaneGF`: Ground plane Green's function instance
- `r_obs::AbstractVector`: Observation point (3D)
- `r_src::AbstractVector`: Source point (3D)

# Returns
- `GreenFuncVals`: Contains g_A (vector potential) and g_phi (scalar potential)

# Notes
- Both potentials subtract the image for horizontal source currents
- Vector potential: tangential E must vanish on ground → image current reversed
- Scalar potential: horizontal dipole → image charge opposite sign
"""
function evaluate_greenfunc(gf::GroundPlaneGF{FT}, 
                             r_obs::AbstractVector, 
                             r_src::AbstractVector) where {FT}
    # Direct contribution
    R_direct = norm(r_obs - r_src)
    G_direct = exp(-im * gf.k * R_direct) / R_direct
    
    # Image contribution
    r_src_image = mirror_point_across_ground(r_src, gf.z_gnd)
    R_image = norm(r_obs - r_src_image)
    G_image = exp(-im * gf.k * R_image) / R_image
    
    # Both vector (A) and scalar (φ) potentials subtract image for horizontal sources
    g_A = G_direct - G_image
    g_phi = G_direct - G_image
    
    return GreenFuncVals{Complex{FT}}(g_A, g_phi)
end

function evaluate_greenfunc(gf::GroundPlaneGF{FT}, 
                             r_obs::Vec3D{FT}, 
                             r_src::Vec3D{FT}) where {FT}
    # Direct contribution using efficient dist function
    R_direct = norm(r_obs - r_src)
    G_direct = exp(-im * gf.k * R_direct) / R_direct
    
    # Image contribution
    r_src_image = mirror_point_across_ground(r_src, gf.z_gnd)
    R_image = norm(r_obs - r_src_image)
    G_image = exp(-im * gf.k * R_image) / R_image
    
    # Both vector and scalar potentials subtract image for horizontal sources
    g_A = G_direct - G_image
    g_phi = G_direct - G_image
    
    return GreenFuncVals{Complex{FT}}(g_A, g_phi)
end

# =============================================================================
# Component-Specific Evaluation
# =============================================================================

"""
    evaluate_greenfunc_direct(gf::GroundPlaneGF, r_obs, r_src)

Evaluate only the direct (free-space) component.

# Returns
- `Complex`: G_direct = exp(-jkR)/R
"""
function evaluate_greenfunc_direct(gf::GroundPlaneGF{FT}, 
                                    r_obs::Vec3D{FT}, 
                                    r_src::Vec3D{FT}) where {FT}
    R_direct = norm(r_obs - r_src)
    return exp(-im * gf.k * R_direct) / R_direct
end

"""
    evaluate_greenfunc_image(gf::GroundPlaneGF, r_obs, r_src)

Evaluate only the image component.

# Returns
- `Complex`: G_image = exp(-jkR_image)/R_image
"""
function evaluate_greenfunc_image(gf::GroundPlaneGF{FT}, 
                                   r_obs::Vec3D{FT}, 
                                   r_src::Vec3D{FT}) where {FT}
    r_src_image = mirror_point_across_ground(r_src, gf.z_gnd)
    R_image = norm(r_obs - r_src_image)
    return exp(-im * gf.k * R_image) / R_image
end

"""
    evaluate_greenfunc_A(gf::GroundPlaneGF, r_obs, r_src)

Evaluate only the vector potential component.

# Mathematical Formulation
```
G_A = G_direct - G_image
```

# Returns
- `Complex`: Vector potential Green's function
"""
function evaluate_greenfunc_A(gf::GroundPlaneGF{FT}, 
                               r_obs::Vec3D{FT}, 
                               r_src::Vec3D{FT}) where {FT}
    G_direct = evaluate_greenfunc_direct(gf, r_obs, r_src)
    G_image = evaluate_greenfunc_image(gf, r_obs, r_src)
    return G_direct - G_image
end

"""
    evaluate_greenfunc_phi(gf::GroundPlaneGF, r_obs, r_src)

Evaluate only the scalar potential component.

# Mathematical Formulation
```
    G_φ = G_direct - G_image
    ```
    
    # Returns
    - `Complex`: Scalar potential Green's function
    """
    function evaluate_greenfunc_phi(gf::GroundPlaneGF{FT}, 
                                     r_obs::Vec3D{FT}, 
                                     r_src::Vec3D{FT}) where {FT}
        G_direct = evaluate_greenfunc_direct(gf, r_obs, r_src)
        G_image = evaluate_greenfunc_image(gf, r_obs, r_src)
        return G_direct - G_image
    end

# =============================================================================
# Singularity Handling
# =============================================================================

"""
    evaluate_greenfunc_star(gf::GroundPlaneGF, r_obs::Vec3D, r_src::Vec3D)

Evaluate de-singularized Green's function with ground plane.

# Mathematical Formulation
For coincident triangles, we extract the 1/R singularity:
```
g*_A   = (G_direct - 1/R_direct) - G_image
g*_φ   = (G_direct - 1/R_direct) - G_image
```

Both vector and scalar potentials subtract the image term for horizontal 
source currents. The image term G_image is smooth (no singularity) because 
the image point is never at the observation point when z > z_gnd.

# Returns
- `GreenFuncVals`: De-singularized values

# Notes
- Used for self-term (coincident triangle) integrals
- Direct term uses Taylor expansion (greenfunc_star_freespace)
- Image term is evaluated directly (no singularity)
- Both g_A and g_phi use the same sign (-) for the image contribution
  when the source current is horizontal (standard MoM surface formulation)
"""
function evaluate_greenfunc_star(gf::GroundPlaneGF{FT}, 
                                  r_obs::Vec3D{FT}, 
                                  r_src::Vec3D{FT};
                                  taylor_order::Int=15) where {FT}
    # Direct term: de-singularized
    R_direct = norm(r_obs - r_src)
    g_star_direct = greenfunc_star_freespace(R_direct, real(gf.k); taylor_order=taylor_order)
    
    # Image term: evaluate directly (no singularity)
    G_image = evaluate_greenfunc_image(gf, r_obs, r_src)
    
    # Combine with appropriate signs
    # For horizontal currents (Phase 1): both A and φ subtract the image
    g_A_star = g_star_direct - G_image      # Vector potential
    g_phi_star = g_star_direct - G_image    # Scalar potential (same sign as g_A for horizontal currents)
    
    return GreenFuncVals{Complex{FT}}(g_A_star, g_phi_star)
end

"""
    is_image_singular(gf::GroundPlaneGF, r_obs::Vec3D, r_src::Vec3D) -> Bool

Check if image contribution becomes singular.

# Returns
- `true` if r_obs is exactly above r_src (R_image = 2*(z - z_gnd))
- `false` otherwise

# Notes
- Image singularity only occurs if r_obs is on ground plane (z = z_gnd)
- For normal operation (z > z_gnd), image is always non-singular
"""
function is_image_singular(gf::GroundPlaneGF{FT}, r_obs::Vec3D{FT}, r_src::Vec3D{FT}) where {FT}
    # Check if observation point is on ground plane
    z_obs = r_obs[3]
    return abs(z_obs - gf.z_gnd) < eps(FT)
end

# =============================================================================
# Geometry Queries
# =============================================================================

"""
    get_ground_plane_z(gf::GroundPlaneGF) -> Real

Get the ground plane z-coordinate.
"""
get_ground_plane_z(gf::GroundPlaneGF) = gf.z_gnd

"""
    height_above_ground(gf::GroundPlaneGF, r::AbstractVector) -> Real

Get height of point above ground plane.

# Returns
- `h = z - z_gnd` (must be >= 0 for valid operation)
"""
height_above_ground(gf::GroundPlaneGF, r::AbstractVector) = r[3] - gf.z_gnd

"""
    is_above_ground(gf::GroundPlaneGF, r::AbstractVector) -> Bool

Check if point is above (or on) the ground plane.
"""
is_above_ground(gf::GroundPlaneGF, r::AbstractVector) = r[3] >= gf.z_gnd

"""
    validate_above_ground(gf::GroundPlaneGF, r_obs, r_src)

Validate that both points are above ground plane.

# Throws
- Error if either point is below ground plane

# Notes
- Image theory requires all sources/observers above ground
- For sources inside ground, different formulation needed
"""
function validate_above_ground(gf::GroundPlaneGF, r_obs, r_src)
    if !is_above_ground(gf, r_obs)
        error("Observation point below ground plane: z=$(r_obs[3]) < z_gnd=$(gf.z_gnd)")
    end
    if !is_above_ground(gf, r_src)
        error("Source point below ground plane: z=$(r_src[3]) < z_gnd=$(gf.z_gnd)")
    end
    return true
end

# =============================================================================
# Special Cases
# =============================================================================

"""
    evaluate_at_same_height(gf::GroundPlaneGF, ρ::Real, h::Real)

Efficient evaluation when source and observer are at same height h above ground.

# Mathematical Formulation
```
ρ = horizontal distance
h = height above ground

R_direct = sqrt(ρ²)
R_image  = sqrt(ρ² + (2h)²)
```

# Arguments
- `gf::GroundPlaneGF`: Ground plane Green's function
- `ρ::Real`: Horizontal separation
- `h::Real`: Height above ground (same for both points)

# Returns
- `GreenFuncVals`: g_A and g_phi

# Notes
- More efficient for same-layer interactions (common case)
- Avoids computing z-coordinates separately
"""
function evaluate_at_same_height(gf::GroundPlaneGF{FT}, ρ::FT, h::FT) where {FT}
    # Direct distance
    R_direct = ρ
    G_direct = exp(-im * gf.k * R_direct) / R_direct
    
    # Image distance (path via ground reflection)
    R_image = sqrt(ρ^2 + (2*h)^2)
    G_image = exp(-im * gf.k * R_image) / R_image
    
    g_A = G_direct - G_image
    g_phi = G_direct - G_image
    
    return GreenFuncVals{Complex{FT}}(g_A, g_phi)
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    wavenumber(gf::GroundPlaneGF) -> Complex

Get the wavenumber.
"""
wavenumber(gf::GroundPlaneGF) = gf.k

"""
    wavelength(gf::GroundPlaneGF) -> Real

Get the wavelength.
"""
wavelength(gf::GroundPlaneGF{FT}) where {FT} = 2π / abs(real(gf.k))

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, gf::GroundPlaneGF{FT}) where {FT} = 
    print(io, "GroundPlaneGF{$FT}(z_gnd=$(gf.z_gnd), k=$(gf.k))")

Base.summary(io::IO, gf::GroundPlaneGF{FT}) where {FT} = 
    print(io, "GroundPlaneGF{$FT} (z_gnd=$(gf.z_gnd), λ=$(wavelength(gf)))")
