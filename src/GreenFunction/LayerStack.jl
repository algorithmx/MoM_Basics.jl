# LayerStack.jl
# Data structures for defining layered media configurations
# Used by LayeredMediaGF for dielectric substrate modeling

# =============================================================================
# Layer Information
# =============================================================================

"""
    LayerInfo{FT<:AbstractFloat}

Information about a single layer in the stack.

Fields:
- `name::String`: Layer identifier (e.g., "substrate", "M1", "air")
- `eps_r::Complex{FT}`: Relative permittivity (complex for lossy dielectrics)
- `mu_r::Complex{FT}`: Relative permeability (usually 1.0)
- `thickness::FT`: Layer thickness in meters (Inf for half-spaces)
- `sigma::FT`: Conductivity in S/m (alternative to imaginary part of eps_r)

# Notes
- Layers are stored bottom to top (increasing z)
- PEC ground planes are represented at the stack level as an interface boundary,
  not as a zero-thickness `LayerInfo`
- For half-spaces (unbounded), set thickness = Inf
- The real part of eps_r accounts for polarization, imaginary part for losses

# Example
```julia
# Silicon substrate
LayerInfo("Si", 11.7, 1.0, 500e-6, 0.0)

# Lossy dielectric
LayerInfo("lossy", 4.3 - 0.1im, 1.0, 1e-3, 0.0)

# Finite-thickness conductor layer
LayerInfo("M1", 1.0, 1.0, 0.35e-6, 3.5e7)

# Air half-space (top)
LayerInfo("air", 1.0, 1.0, Inf, 0.0)
```
"""
struct LayerInfo{FT<:AbstractFloat}
    name::String
    eps_r::Complex{FT}
    mu_r::Complex{FT}
    thickness::FT
    sigma::FT
    
    function LayerInfo{FT}(name::String, 
                           eps_r::Complex{FT}, 
                           mu_r::Complex{FT}, 
                           thickness::FT,
                           sigma::FT = zero(FT)) where {FT<:AbstractFloat}
        # Validate thickness
        if thickness <= 0 && !isinf(thickness)
            error("Layer thickness must be positive or Inf, got $thickness")
        end
        new(name, eps_r, mu_r, thickness, sigma)
    end
end

# Convenience constructors
LayerInfo(name::String, eps_r::Complex{FT}, mu_r::Complex{FT}, thickness::FT, sigma::FT=zero(FT)) where {FT} =
    LayerInfo{FT}(name, eps_r, mu_r, thickness, sigma)

LayerInfo(name::String, eps_r::Real, mu_r::Real, thickness::Real, sigma::Real=0.0) =
    LayerInfo(name, Complex(eps_r), Complex(mu_r), thickness, sigma)

# Handle case where eps_r is complex but mu_r is real (from test)
LayerInfo(name::String, eps_r::Complex, mu_r::Real, thickness::Real, sigma::Real=0.0) =
    LayerInfo(name, eps_r, Complex(mu_r), thickness, sigma)

"""
    is_halfspace(layer::LayerInfo) -> Bool

Check if layer is a half-space (unbounded).
"""
is_halfspace(layer::LayerInfo) = isinf(layer.thickness)

"""
    is_pec(layer::LayerInfo) -> Bool

Check if layer represents a PEC conductor.
"""
function is_pec(layer::LayerInfo{FT}) where {FT}
    # PEC is modeled as infinite conductivity or very large value
    return layer.sigma > FT(1e8) || abs(layer.eps_r) > FT(1e6)
end

"""
    effective_permittivity(layer::LayerInfo, frequency::Real) -> Complex

Compute effective complex permittivity at given frequency.

Accounts for conductivity: eps_eff = eps' - j*(eps'' + sigma/omega)
"""
function effective_permittivity(layer::LayerInfo{FT}, frequency::Real) where {FT}
    ω = 2π * frequency
    ε₀ = FT(8.854187817e-12)  # Vacuum permittivity
    eps_r = layer.eps_r
    
    # Add conductivity contribution to imaginary part
    if layer.sigma > 0
        eps_eff = real(eps_r) - imag(eps_r)*im - im * layer.sigma / (ω * ε₀)
        return eps_eff
    else
        return eps_r
    end
end

# =============================================================================
# Layer Stack
# =============================================================================

"""
    LayerStack{FT<:AbstractFloat}

Complete stack of layers defining the medium.

Fields:
- `layers::Vector{LayerInfo{FT}}`: Layers from bottom to top
- `interfaces::Vector{FT}`: z-coordinates of layer interfaces
- `n_layers::Int`: Total number of layers
- `reference_z::FT`: Reference z-coordinate (usually ground plane at 0)
- `has_ground_plane::Bool`: Whether a PEC ground boundary exists at `reference_z`

# Notes
- Layer i occupies: interfaces[i] < z < interfaces[i+1]
- interfaces[1] is the bottom boundary
- interfaces[end] is the top boundary (or +Inf)
- Source and observation points must be located within layers
- When `has_ground_plane=true`, `reference_z` is the PEC ground interface below the
    first material layer

# Example: Microstrip on grounded substrate
```julia
layers = [
    LayerInfo("substrate", 11.7, 1.0, 500e-6), # Silicon substrate
    LayerInfo("air", 1.0, 1.0, Inf)            # Air half-space
]
stack = LayerStack(layers; reference_z=0.0, has_ground_plane=true)
```
"""
struct LayerStack{FT<:AbstractFloat}
    layers::Vector{LayerInfo{FT}}
    interfaces::Vector{FT}
    n_layers::Int
    reference_z::FT
    has_ground_plane::Bool
    
    function LayerStack{FT}(layers::Vector{LayerInfo{FT}};
                            reference_z::FT=zero(FT),
                            has_ground_plane::Bool=false) where {FT<:AbstractFloat}
        n = length(layers)
        n >= 1 || error("LayerStack requires at least one material layer")
        
        # Compute interface positions
        interfaces = Vector{FT}(undef, n+1)
        interfaces[1] = reference_z
        for i in 1:n
            if is_halfspace(layers[i])
                # Half-space extends to infinity
                interfaces[i+1] = (i == n) ? FT(Inf) : error("Only top layer can be half-space")
            else
                interfaces[i+1] = interfaces[i] + layers[i].thickness
            end
        end
        
        new(layers, interfaces, n, reference_z, has_ground_plane)
    end
end

# Convenience constructor
LayerStack(layers::Vector{LayerInfo{FT}}; reference_z::FT=zero(FT), has_ground_plane::Bool=false) where {FT} =
    LayerStack{FT}(layers; reference_z=reference_z, has_ground_plane=has_ground_plane)

"""
    has_ground_plane(stack::LayerStack) -> Bool

Check whether the stack has a PEC ground boundary at `reference_z`.
"""
has_ground_plane(stack::LayerStack) = stack.has_ground_plane

"""
    ground_plane_z(stack::LayerStack) -> Real

Get the PEC ground boundary location when `has_ground_plane(stack)` is true.
"""
ground_plane_z(stack::LayerStack) = stack.reference_z

"""
    get_layer_index(stack::LayerStack, z::Real) -> Int

Determine which layer contains the given z-coordinate.

# Returns
- `Int`: Layer index (1-indexed)

# Errors
- If z is outside valid range
"""
function get_layer_index(stack::LayerStack{FT}, z::Real) where {FT}
    z_FT = FT(z)
    for i in 1:stack.n_layers
        if stack.interfaces[i] <= z_FT < stack.interfaces[i+1]
            return i
        end
    end
    # Check if exactly at top boundary of last layer
    if z_FT == stack.interfaces[end]
        return stack.n_layers
    end
    error("z = $z is outside the layer stack range")
end

"""
    get_layer(stack::LayerStack, z::Real) -> LayerInfo

Get the layer containing the given z-coordinate.
"""
get_layer(stack::LayerStack, z::Real) = stack.layers[get_layer_index(stack, z)]

"""
    layer_thickness_above(stack::LayerStack, z::Real) -> Real

Get distance from point z to top of its layer.
"""
function layer_thickness_above(stack::LayerStack, z::Real)
    idx = get_layer_index(stack, z)
    return stack.interfaces[idx+1] - z
end

"""
    layer_thickness_below(stack::LayerStack, z::Real) -> Real

Get distance from point z to bottom of its layer.
"""
function layer_thickness_below(stack::LayerStack, z::Real)
    idx = get_layer_index(stack, z)
    return z - stack.interfaces[idx]
end

"""
    are_in_same_layer(stack::LayerStack, z1::Real, z2::Real) -> Bool

Check if two z-coordinates are in the same layer.
"""
are_in_same_layer(stack::LayerStack, z1::Real, z2::Real) = 
    get_layer_index(stack, z1) == get_layer_index(stack, z2)

# =============================================================================
# JSON I/O (Placeholder for future implementation)
# =============================================================================

"""
    load_layer_stack(filename::String) -> LayerStack

Load layer stack definition from JSON file.

# Placeholder
This function will be fully implemented in Phase 2.
For now, it creates a default test stack.
"""
function load_layer_stack(filename::String)
    # PLACEHOLDER: Will implement JSON parsing in Phase 2
    # For now, create a default grounded microstrip stack for testing.
    # The PEC ground is represented as a boundary at reference_z, not as a layer.
    FT = Precision.FT
    
    @warn "JSON loading not yet implemented. Using default test stack."
    
    layers = [
        LayerInfo("substrate", Complex{FT}(11.7), Complex{FT}(1.0), FT(500e-6)),
        LayerInfo("air", Complex{FT}(1.0), Complex{FT}(1.0), FT(Inf))
    ]
    
    return LayerStack{FT}(layers; reference_z=zero(FT), has_ground_plane=true)
end

"""
    save_layer_stack(stack::LayerStack, filename::String)

Save layer stack definition to JSON file.

# Placeholder
This function will be fully implemented in Phase 2.
"""
function save_layer_stack(stack::LayerStack, filename::String)
    # PLACEHOLDER: Will implement JSON writing in Phase 2
    @warn "JSON saving not yet implemented. Stack not saved."
    return nothing
end

# =============================================================================
# DCIM Coefficients (for LayeredMediaGF - placeholder)
# =============================================================================

"""
    ComplexImage{FT<:AbstractFloat}

Represents a single complex image in DCIM expansion.

Fields:
- `amplitude::Complex{FT}`: Complex amplitude (residue) a_i
- `distance::Complex{FT}`: Complex distance (pole) b_i

# Notes
- The spatial Green's function is approximated as:
  G(ρ, z, z') ≈ Σ a_i * exp(-jk*R_i) / (4π*R_i)
  where R_i = sqrt(ρ² + (z-z')² - b_i²)
"""
struct ComplexImage{FT<:AbstractFloat}
    amplitude::Complex{FT}
    distance::Complex{FT}
end

"""
    DCIMCoefficients{FT<:AbstractFloat}

Precomputed DCIM coefficients for layered media Green's function.

Fields:
- `quasi_static::Vector{ComplexImage{FT}}`: Quasi-static images (frequency-independent)
- `level1::Vector{ComplexImage{FT}}`: Near-field complex images
- `level2::Vector{ComplexImage{FT}}`: Far-field complex images
- `sw_poles::Vector{Complex{FT}}`: Surface wave poles (optional)

# Notes
- Two-level DCIM (Aksun 1996) uses separate sampling for near and far field
- Quasi-static images handle the 1/R singularity and near-field behavior
- Surface wave poles are extracted for improved far-field accuracy
"""
struct DCIMCoefficients{FT<:AbstractFloat}
    quasi_static::Vector{ComplexImage{FT}}
    level1::Vector{ComplexImage{FT}}
    level2::Vector{ComplexImage{FT}}
    sw_poles::Vector{Complex{FT}}
    
    # Empty constructor
    function DCIMCoefficients{FT}() where {FT<:AbstractFloat}
        new(ComplexImage{FT}[], ComplexImage{FT}[], ComplexImage{FT}[], Complex{FT}[])
    end
    
    # Full constructor
    function DCIMCoefficients{FT}(quasi_static::Vector{ComplexImage{FT}},
                                   level1::Vector{ComplexImage{FT}},
                                   level2::Vector{ComplexImage{FT}},
                                   sw_poles::Vector{Complex{FT}}) where {FT<:AbstractFloat}
        new(quasi_static, level1, level2, sw_poles)
    end
end

# Convenience constructor
DCIMCoefficients(quasi_static::Vector{ComplexImage{FT}},
                  level1::Vector{ComplexImage{FT}},
                  level2::Vector{ComplexImage{FT}},
                  sw_poles::Vector{Complex{FT}}) where {FT} =
    DCIMCoefficients{FT}(quasi_static, level1, level2, sw_poles)

# Placeholder for DCIM computation (Phase 3)
"""
    compute_dcim_coefficients(stack::LayerStack, frequency::Real, 
                              source_layer::Int, obs_layer::Int) -> DCIMCoefficients

Compute DCIM coefficients for given source-observer layer pair.

# Placeholder
Will be implemented in Phase 3 with GPOF fitting.
"""
function compute_dcim_coefficients(stack::LayerStack{FT}, frequency::Real,
                                    source_layer::Int, obs_layer::Int) where {FT}
    # PLACEHOLDER: Will implement two-level DCIM with GPOF in Phase 3
    coeffs = DCIMCoefficients{FT}()
    
    # For now, return empty coefficients
    @warn "DCIM computation not yet implemented. Returning empty coefficients."
    
    return coeffs
end
