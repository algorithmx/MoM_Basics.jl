# LayeredMediaGF.jl
# Layered Media Green's Function via Discrete Complex Image Method (DCIM)
#
# This file implements the full layered media Green's function using:
# - Spectral domain computation via transmission line recursion
# - Two-level DCIM for complex image extraction
# - Spatial evaluation via complex images
#
# Reference:
# - Aksun, IEEE T-MTT, 1996
# - Michalski & Mosig, IEEE T-AP, 1997

# =============================================================================
# Layered Media Green's Function Type
# =============================================================================

"""
    LayeredMediaGF{FT<:AbstractFloat} <: AbstractGreenFunction{FT}

Layered media Green's function using DCIM (Discrete Complex Image Method).

# Architecture

    LayerStack  →  SpectralGF (TL recursion)  →  DCIM (GPOF fitting)  →  evaluate_dcim
    ──────────     ─────────────────────────     ──────────────────      ──────────────
    εr, μr, d      G̃_A(k_ρ), G̃_φ(k_ρ)          complex images         G(ρ) = Σ aᵢ e^{-jkRᵢ}/Rᵢ

The constructor calls `compute_all_dcim_coefficients!` which runs the full
spectral GF → DCIM pipeline for every (source, observer) layer pair.
Subsequent `evaluate_greenfunc` calls are fast (just summing complex images).

# Convention
G(R) = exp(−jkR)/R  (NO 1/(4π) factor — the MoM kernel applies it externally).

# Fields
- `stack::LayerStack{FT}`: Layer geometry (thicknesses, εr, μr, ground plane)
- `frequency::FT`: Operating frequency (Hz)
- `k::Complex{FT}`: Free-space wavenumber k₀ = ω/c
- `g_a_coeffs`: Dict mapping (src, obs) layer pair → DCIM coefficients for G_A
- `g_phi_coeffs`: Dict mapping (src, obs) layer pair → DCIM coefficients for G_φ
- `dcim_params::DCIMParameters{FT}`: DCIM fitting parameters (T01, T02, N, etc.)

# Usage
```julia
layers = [
    LayerInfo("substrate", 11.7, 1.0, 500e-6),
    LayerInfo("air", 1.0, 1.0, Inf)
]
stack = LayerStack(layers; reference_z=0.0, has_ground_plane=true)
gf = LayeredMediaGF(stack, 4e9)  # constructs and fits DCIM automatically
vals = evaluate_greenfunc(gf, r_obs, r_src)  # fast evaluation
```

# References
- Aksun, IEEE T-MTT, vol. 44, pp. 651-658, 1996
- Michalski & Mosig, IEEE T-AP, vol. 45, pp. 508-519, 1997
"""
mutable struct LayeredMediaGF{FT<:AbstractFloat} <: AbstractGreenFunction{FT}
    stack::LayerStack{FT}
    frequency::FT
    k::Complex{FT}
    
    # DCIM coefficients for each layer pair (source_layer, obs_layer)
    g_a_coeffs::Dict{Tuple{Int,Int}, DCIMCoefficients{FT}}
    g_phi_coeffs::Dict{Tuple{Int,Int}, DCIMCoefficients{FT}}
    
    # DCIM parameters used for fitting
    dcim_params::DCIMParameters{FT}
    
    function LayeredMediaGF{FT}(stack::LayerStack{FT}, frequency::Real;
                                 dcim_params::DCIMParameters{FT}=DCIMParameters{FT}()) where {FT<:AbstractFloat}
        k = 2π * frequency / FT(299792458.0)  # ω/c
        
        # Initialize coefficient dictionaries
        g_a_coeffs = Dict{Tuple{Int,Int}, DCIMCoefficients{FT}}()
        g_phi_coeffs = Dict{Tuple{Int,Int}, DCIMCoefficients{FT}}()
        
        gf = new(stack, FT(frequency), Complex{FT}(k), g_a_coeffs, g_phi_coeffs, dcim_params)
        
        # Compute DCIM coefficients for all layer pairs
        compute_all_dcim_coefficients!(gf)
        
        return gf
    end
end

# Convenience constructor
LayeredMediaGF(stack::LayerStack{FT}, frequency::Real; 
               dcim_params::DCIMParameters{FT}=DCIMParameters{FT}()) where {FT} = 
    LayeredMediaGF{FT}(stack, frequency; dcim_params=dcim_params)

# =============================================================================
# DCIM Coefficients Computation
# =============================================================================

"""
    compute_all_dcim_coefficients!(gf::LayeredMediaGF)

Compute DCIM coefficients for all (source_layer, obs_layer) pairs.

For each pair, the pipeline is:
1. `dcim_fit` samples the spectral GF G̃(kρ) on two complex-kz paths,
   extracts F = kz × G̃, subtracts the free-space asymptote, fits with
   GPOF, and returns complex image amplitudes + distances.
2. g_A uses **TE** reflection coefficients (vector potential for
   horizontal currents).
3. g_φ uses **TM** reflection coefficients (scalar potential).
4. Results are stored in `gf.g_a_coeffs` and `gf.g_phi_coeffs` dicts,
   keyed by `(src_layer, obs_layer)` tuple.
5. The free-space wavenumber k₀ (not the substrate k) is passed to
   `dcim_fit` because the DCIM fitting paths are parameterized in
   the free-space vertical wavenumber k_{z0}.
"""
function compute_all_dcim_coefficients!(gf::LayeredMediaGF{FT}) where {FT}
    n_layers = gf.stack.n_layers
    k0 = FT(abs(gf.k))
    
    @info "Computing DCIM coefficients for $(n_layers)×$(n_layers) layer pairs..."
    
    for src_layer in 1:n_layers
        for obs_layer in 1:n_layers
            # Determine representative z position for source in this layer
            z_bot = gf.stack.interfaces[src_layer]
            z_top = gf.stack.interfaces[src_layer + 1]
            if isinf(z_top)
                z_src = z_bot + FT(1e-6)
            else
                z_src = (z_bot + z_top) / 2
            end
            
            # Apply DCIM fitting for g_A (TE mode)
            dcim_result_A = dcim_fit(gf.stack, gf.frequency, src_layer, obs_layer, k0;
                                      params=gf.dcim_params, field_type=:A, z_source=z_src)
            
            # Apply DCIM fitting for g_phi (TM mode)
            dcim_result_phi = dcim_fit(gf.stack, gf.frequency, src_layer, obs_layer, k0;
                                        params=gf.dcim_params, field_type=:phi, z_source=z_src)
            
            # Convert DCIMFittingResult to DCIMCoefficients
            gf.g_a_coeffs[(src_layer, obs_layer)] = DCIMCoefficients{FT}(
                dcim_result_A.quasi_static,
                dcim_result_A.level1,
                dcim_result_A.level2,
                dcim_result_A.sw_poles
            )
            
            gf.g_phi_coeffs[(src_layer, obs_layer)] = DCIMCoefficients{FT}(
                dcim_result_phi.quasi_static,
                dcim_result_phi.level1,
                dcim_result_phi.level2,
                dcim_result_phi.sw_poles
            )
        end
    end
    
    total_images = sum(length(gf.g_a_coeffs[key].level1) + 
                       length(gf.g_a_coeffs[key].level2) for key in keys(gf.g_a_coeffs))
    @info "DCIM computation complete. Total complex images: $total_images"
    
    return nothing
end

"""
    generate_k_rho_samples(k0::FT, params::DCIMParameters{FT}) where {FT}

Generate k_ρ samples for spectral GF computation.
(Legacy helper — not used by the new two-level DCIM which generates samples internally.)
"""
function generate_k_rho_samples(k0::FT, params::DCIMParameters{FT}) where {FT}
    # Use T01 and n_samples from the new parameter layout
    T01 = params.T01
    N = params.n_samples
    
    # Sample along the imaginary kz axis: kρ = k0 * sqrt(1 + t²), t ∈ [0, T01]
    dt = T01 / (N - 1)
    k_rho_all = FT[k0 * sqrt(one(FT) + (n * dt)^2) for n in 0:(N-1)]
    
    return k_rho_all
end

# =============================================================================
# Core Evaluation Functions
# =============================================================================

"""
    evaluate_greenfunc(gf::LayeredMediaGF, r_obs, r_src)

Evaluate layered media Green's function at observation and source points.

Computes G_A(ρ,z,z') and G_φ(ρ,z,z') from precomputed DCIM coefficients:

    G(ρ) = Σ aᵢ exp(−jk₀ Rᵢ) / Rᵢ

where:
- ρ = horizontal distance = √((x−x')² + (y−y')²)
- Rᵢ = complex distance from DCIM fitting (see `evaluate_dcim`)
- k₀ = `gf.k` = free-space wavenumber ω/c
- aᵢ already includes the 2j convention factor

The free-space k₀ is used (not the substrate wavenumber) because the
DCIM complex images are parameterized via the Sommerfeld identity in
terms of the free-space vertical wavenumber k_{z0}.

Convention: G = exp(−jkR)/R (no 1/(4π) factor; the MoM kernel
applies 1/(4π) externally).
"""
function evaluate_greenfunc(gf::LayeredMediaGF{FT}, 
                             r_obs::AbstractVector, 
                             r_src::AbstractVector) where {FT}
    
    # Compute horizontal distance ρ
    ρ = horizontal_distance(r_obs, r_src)
    
    # Get z-coordinates
    z_obs = r_obs[3]
    z_src = r_src[3]
    
    # Determine layer indices
    src_layer = get_layer_index(gf.stack, z_src)
    obs_layer = get_layer_index(gf.stack, z_obs)
    
    # Get DCIM coefficients for this layer pair
    key = (src_layer, obs_layer)
    coeffs_A = gf.g_a_coeffs[key]
    coeffs_phi = gf.g_phi_coeffs[key]
    
    # Use FREE-SPACE k₀ for evaluate_dcim.
    # Why k₀ and not k_substrate?
    # The DCIM fitting paths (Level 1 & 2) are parameterized using k₀, and the
    # complex image distances α are defined so that exp(−α kz₀) matches the
    # spectral data.  The Sommerfeld identity that maps each complex image to
    # exp(−jkR)/R in spatial domain uses the SAME k₀.
    g_A = evaluate_dcim(coeffs_A, ρ, z_obs, z_src, gf.k)
    g_phi = evaluate_dcim(coeffs_phi, ρ, z_obs, z_src, gf.k)
    
    return GreenFuncVals{Complex{FT}}(g_A, g_phi)
end

"""
    evaluate_greenfunc(gf::LayeredMediaGF, r_obs::Vec3D{FT}, r_src::Vec3D{FT}) where {FT}

Optimized evaluation for Vec3D types.
"""
function evaluate_greenfunc(gf::LayeredMediaGF{FT}, 
                             r_obs::Vec3D{FT}, 
                             r_src::Vec3D{FT}) where {FT}
    # Compute horizontal distance ρ
    ρ = sqrt((r_obs[1]-r_src[1])^2 + (r_obs[2]-r_src[2])^2)
    
    # Get z-coordinates
    z_obs = r_obs[3]
    z_src = r_src[3]
    
    # Determine layer indices
    src_layer = get_layer_index(gf.stack, z_src)
    obs_layer = get_layer_index(gf.stack, z_obs)
    
    # Get DCIM coefficients
    key = (src_layer, obs_layer)
    coeffs_A = gf.g_a_coeffs[key]
    coeffs_phi = gf.g_phi_coeffs[key]
    
    # Use FREE-SPACE k₀ (see note in AbstractVector overload above)
    g_A = evaluate_dcim(coeffs_A, ρ, z_obs, z_src, gf.k)
    g_phi = evaluate_dcim(coeffs_phi, ρ, z_obs, z_src, gf.k)
    
    return GreenFuncVals{Complex{FT}}(g_A, g_phi)
end

"""
    evaluate_at_same_height(gf::LayeredMediaGF, ρ, h, layer_idx)

Optimized evaluation when source and observer are at the same height z.
Common in microstrip MoM where all conductors sit on the substrate surface.
Skips the layer-index lookup since the caller provides it directly.
"""
function evaluate_at_same_height(gf::LayeredMediaGF{FT}, ρ::FT, h::FT, layer_idx::Int) where {FT}
    key = (layer_idx, layer_idx)
    coeffs_A = gf.g_a_coeffs[key]
    coeffs_phi = gf.g_phi_coeffs[key]
    
    z = gf.stack.interfaces[1] + h  # reference_z + h
    
    g_A = evaluate_dcim(coeffs_A, ρ, z, z, gf.k)
    g_phi = evaluate_dcim(coeffs_phi, ρ, z, z, gf.k)
    
    return GreenFuncVals{Complex{FT}}(g_A, g_phi)
end

# =============================================================================
# Singularity Handling
# =============================================================================

"""
    evaluate_greenfunc_star(gf::LayeredMediaGF, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate de-singularized Green's function for layered media.

# Mathematical Formulation

For self-term (coincident triangle) integrals in MoM, the 1/R singularity is
extracted analytically. This function returns the smooth part of the layered
media Green's function, excluding the quasi-static images which contain the
singularity.

The DCIM expansion consists of:
- **Quasi-static images**: Contain the 1/R singularity (handled separately)
- **Level 1 & 2 complex images**: Smooth corrections (returned by this function)
- **Surface wave poles**: Smooth far-field contribution (returned by this function)

The de-singularized Green's function is:
```
g*_A = Σ_Level1,Level2 aᵢ exp(−jkRᵢ) / Rᵢ    (smooth part only)
g*_φ = Σ_Level1,Level2 aᵢ exp(−jkRᵢ) / Rᵢ    (smooth part only)
```

# Arguments
- `gf::LayeredMediaGF`: Layered media Green's function instance
- `r_obs::AbstractVector`: Observation point (3D)
- `r_src::AbstractVector`: Source point (3D)

# Returns
- `GreenFuncVals`: Contains g_A_star and g_phi_star (both singularities removed)

# Notes
- Used for self-term (coincident triangle) integrals in MoM matrix assembly
- The singular part (quasi-static images) is handled by `greenfunc_star_freespace`
- The total Green's function is: G = G_singular + G_star
- For layered media, G_singular includes the free-space direct term and 
  quasi-static image contributions from dielectric interfaces

# References
- Simsek, Liu & Wei, "Singularity Subtraction for Evaluation of Green's 
  Functions for Multilayer Media," IEEE T-MTT, vol. 54, pp. 216-225, 2006.
- Michalski & Mosig, IEEE T-AP, vol. 45, pp. 508-519, 1997.
"""
function evaluate_greenfunc_star(gf::LayeredMediaGF{FT}, 
                                  r_obs::AbstractVector, 
                                  r_src::AbstractVector) where {FT}
    # Compute horizontal distance ρ = sqrt((x-x')² + (y-y')²)
    # Note: For coincident triangles in MoM self-terms, ρ → 0 but we still need
    # the layered media correction which remains finite
    ρ = horizontal_distance(r_obs, r_src)
    
    # Get z-coordinates for vertical separation
    # The z-dependence is crucial for layered media - different from free space
    z_obs = r_obs[3]
    z_src = r_src[3]
    
    # Determine which layers contain source and observer
    # DCIM coefficients are pre-computed per layer pair (src_layer, obs_layer)
    # For same-layer interactions (most common in MoM), src_layer == obs_layer
    src_layer = get_layer_index(gf.stack, z_src)
    obs_layer = get_layer_index(gf.stack, z_obs)
    
    # Retrieve pre-computed DCIM coefficients for this layer pair
    # coeffs_A and coeffs_phi contain complex images from GPOF fitting
    key = (src_layer, obs_layer)
    coeffs_A = gf.g_a_coeffs[key]
    coeffs_phi = gf.g_phi_coeffs[key]
    
    # Evaluate the SMOOTH part of the Green's function
    # 
    # Theory: The DCIM expansion G = G_qs + G_L1 + G_L2 consists of:
    #   G_qs  = quasi-static images (contain 1/R singularity)
    #   G_L1  = Level 1 complex images (smooth, intermediate field)
    #   G_L2  = Level 2 complex images (smooth, far field/surface waves)
    #
    # For MoM self-term integrals, the singular part G_qs is handled separately
    # via analytical integration (greenfunc_star_freespace), while this function
    # returns only the smooth correction G* = G_L1 + G_L2.
    #
    # Reference: Simsek et al., IEEE T-MTT, 2006, Eq. (14)-(16)
    g_A_star = evaluate_dcim_smooth(coeffs_A, ρ, z_obs, z_src, gf.k)
    g_phi_star = evaluate_dcim_smooth(coeffs_phi, ρ, z_obs, z_src, gf.k)
    
    return GreenFuncVals{Complex{FT}}(g_A_star, g_phi_star)
end

function evaluate_greenfunc_star(gf::LayeredMediaGF{FT}, 
                                  r_obs::Vec3D{FT}, 
                                  r_src::Vec3D{FT}) where {FT}
    # Optimized version for Vec3D types
    # Compute horizontal distance ρ = sqrt((x-x')² + (y-y')²)
    ρ = sqrt((r_obs[1]-r_src[1])^2 + (r_obs[2]-r_src[2])^2)
    
    # Get z-coordinates for vertical separation
    z_obs = r_obs[3]
    z_src = r_src[3]
    
    # Determine layer indices for coefficient lookup
    src_layer = get_layer_index(gf.stack, z_src)
    obs_layer = get_layer_index(gf.stack, z_obs)
    
    # Retrieve DCIM coefficients and evaluate smooth part
    key = (src_layer, obs_layer)
    coeffs_A = gf.g_a_coeffs[key]
    coeffs_phi = gf.g_phi_coeffs[key]
    
    # Return only Level 1 + Level 2 complex images (smooth part)
    # Quasi-static images with 1/R singularity are excluded
    g_A_star = evaluate_dcim_smooth(coeffs_A, ρ, z_obs, z_src, gf.k)
    g_phi_star = evaluate_dcim_smooth(coeffs_phi, ρ, z_obs, z_src, gf.k)
    
    return GreenFuncVals{Complex{FT}}(g_A_star, g_phi_star)
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_layer_wavenumber(gf::LayeredMediaGF, layer_idx) -> Complex

Compute the wavenumber for a specific layer: k_i = ω √(μ_i ε_i).

Note: This is the LAYER wavenumber, not the free-space k₀ used by DCIM.
It can be useful for computing substrate wavelength, loss tangent, etc.,
but should NOT be passed to `evaluate_dcim` (which expects k₀).
"""
function compute_layer_wavenumber(gf::LayeredMediaGF{FT}, layer_idx::Int) where {FT}
    ε₀ = FT(8.854187817e-12)
    μ₀ = FT(4π * 1e-7)
    ω = 2π * gf.frequency
    
    layer = gf.stack.layers[layer_idx]
    μ_i = μ₀ * real(layer.mu_r)
    ε_i = ε₀ * effective_permittivity(layer, gf.frequency)
    
    return complex(ω * sqrt(μ_i * ε_i))
end

"""
    get_layer_stack(gf::LayeredMediaGF) -> LayerStack

Get the layer stack associated with this Green's function.
"""
get_layer_stack(gf::LayeredMediaGF) = gf.stack

"""
    get_frequency(gf::LayeredMediaGF) -> Real

Get the operating frequency.
"""
get_frequency(gf::LayeredMediaGF) = gf.frequency

"""
    wavenumber(gf::LayeredMediaGF) -> Complex

Get the free-space wavenumber.
"""
wavenumber(gf::LayeredMediaGF) = gf.k

"""
    wavelength(gf::LayeredMediaGF) -> Real

Get the free-space wavelength.
"""
wavelength(gf::LayeredMediaGF{FT}) where {FT} = 2π / abs(real(gf.k))

"""
    get_dcim_info(gf::LayeredMediaGF) -> Dict

Get diagnostic information about DCIM fitting.

Returns dictionary with:
- `total_layer_pairs`: Number of layer pairs
- `total_complex_images`: Total number of complex images
- `avg_images_per_pair`: Average images per layer pair
"""
function get_dcim_info(gf::LayeredMediaGF)
    info = Dict{String, Any}()
    
    n_pairs = length(gf.g_a_coeffs)
    total_images = 0
    
    for key in keys(gf.g_a_coeffs)
        coeffs = gf.g_a_coeffs[key]
        total_images += length(coeffs.level1) + length(coeffs.level2)
    end
    
    info["total_layer_pairs"] = n_pairs
    info["total_complex_images"] = total_images
    info["avg_images_per_pair"] = n_pairs > 0 ? total_images / n_pairs : 0
    info["dcim_parameters"] = gf.dcim_params
    
    return info
end

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, gf::LayeredMediaGF{FT}) where {FT} = 
    print(io, "LayeredMediaGF{$FT}(freq=$(gf.frequency), layers=$(gf.stack.n_layers))")

Base.summary(io::IO, gf::LayeredMediaGF{FT}) where {FT} = 
    print(io, "LayeredMediaGF{$FT} ($(gf.stack.n_layers) layers, $(gf.frequency/1e9) GHz, DCIM)")
