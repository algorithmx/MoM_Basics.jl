# SpectralGF.jl
# Spectral Domain Green's Function via Transmission Line Recursion
#
# This module computes the spectral Green's function G̃(k_ρ, z, z') for layered
# media using the transmission line analogy.
#
# Reference:
# - K. A. Michalski and J. R. Mosig, "Multilayered media Green's functions in 
#   integral equation formulations," IEEE Trans. Antennas Propagat., vol. 45, 
#   pp. 508-519, March 1997.

# =============================================================================
# Spectral GF Types
# =============================================================================

"""
    SpectralGF{FT<:AbstractFloat}

Spectral domain Green's function container.

Fields:
- `k_rho::Vector{FT}`: Spectral variable samples
- `G_tilde_A::Vector{Complex{FT}}`: Vector potential spectral GF
- `G_tilde_phi::Vector{Complex{FT}}`: Scalar potential spectral GF
- `frequency::FT`: Operating frequency
- `source_layer::Int`: Source layer index
- `obs_layer::Int`: Observer layer index

# Mathematical Formulation
The spectral GF relates to spatial GF via Sommerfeld integral:
```
G(ρ, z, z') = (1/2π) ∫₀^∞ G̃(k_ρ, z, z') J₀(k_ρ ρ) k_ρ dk_ρ
```

For layered media, G̃ is computed via transmission line recursion.
"""
struct SpectralGF{FT<:AbstractFloat}
    k_rho::Vector{FT}
    G_tilde_A::Vector{Complex{FT}}
    G_tilde_phi::Vector{Complex{FT}}
    frequency::FT
    source_layer::Int
    obs_layer::Int
end

"""
    LayerWavenumbers{FT}

Vertical wavenumbers for all layers at a specific k_ρ.

Fields:
- `k_z::Vector{Complex{FT}}`: Vertical wavenumber for each layer
- `k_ρ::FT`: Horizontal wavenumber (spectral variable)
"""
struct LayerWavenumbers{FT<:AbstractFloat}
    k_z::Vector{Complex{FT}}
    k_ρ::FT
end

"""
    TEImpedances{FT}

TE mode characteristic impedances for all layers.

Formula: Z_i^TE = (ω μ_i) / k_{z,i}
"""
struct TEImpedances{FT<:AbstractFloat}
    Z::Vector{Complex{FT}}
end

"""
    TMImpedances{FT}

TM mode characteristic impedances for all layers.

Formula: Z_i^TM = k_{z,i} / (ω ε_i)
"""
struct TMImpedances{FT<:AbstractFloat}
    Z::Vector{Complex{FT}}
end

"""
    ReflectionCoefficients{FT}

Reflection coefficients at layer interfaces.

Fields:
- `Γ_TE::Vector{Complex{FT}}`: TE reflection coefficients
- `Γ_TM::Vector{FT}`: TM reflection coefficients

Note: Vector length is n_layers + 1 (including half-spaces).
"""
struct ReflectionCoefficients{FT<:AbstractFloat}
    Γ_TE::Vector{Complex{FT}}
    Γ_TM::Vector{Complex{FT}}
end

# =============================================================================
# Core Spectral GF Computation
# =============================================================================

"""
    compute_spectral_gf(stack::LayerStack{FT}, frequency::Real,
                        source_layer::Int, obs_layer::Int,
                        k_rho_samples::Vector{FT}) where {FT}

Compute spectral domain Green's function via transmission line recursion.

# Algorithm

For each k_ρ sample:

1. **Vertical wavenumbers**: k_{z,i} = √(k_i² − k_ρ²), branch cut Im(k_z) ≤ 0
2. **Characteristic impedances**: Z^TE = (ωμ)/k_z, Z^TM = k_z/(ωε)
3. **Two-direction TL recursion**:
   - Downward (top → bottom): yields Z_up[i] (impedance looking up)
   - Upward (ground → top): yields Z_down[i] (impedance looking down)
   - With PEC ground: Z_load = 0 at bottom (short circuit, Γ = −1)
4. **Generalized reflection coefficients**: Γ↓[i], Γ↑[i] from impedance mismatch
5. **z-dependent spectral GF** (G = exp(−jkR)/R convention, no 1/(4π)):

   **Vector potential** (TE reflections only; Michalski-Mosig Form C, eq. 42):
   ```
   G̃_A(k_ρ) = 1/(2jk_z) × F^TE
   ```
   **Scalar potential** (TE/TM coupled; Michalski-Mosig Form C, eq. 50):
   ```
   G̃_φ(k_ρ) = −jωε₀/(2k_ρ²) × (Z^TE F^TE − Z^TM F^TM)
   ```
   where F^p = (1 + Γ↓^p e^{-2jk_z h↓})(1 + Γ↑^p e^{-2jk_z h↑}) / D^p.

   The scalar potential couples **both** TE and TM modes because a horizontal
   electric dipole radiates both wave types. In free space (no reflections),
   Z^TE − Z^TM = (ωμ/k_z − k_z/(ωε)) and the formula reduces to 1/(2jk_z)
   so G̃_A = G̃_φ as expected.

# Convention
G̃_A has NO μ/ε prefactors (they are applied by the MoM kernel).
The 1/(2jk_z) is the spectral kernel for our G = exp(−jkR)/R convention.

# References
- Michalski & Mosig, IEEE T-AP, 1997
"""
function compute_spectral_gf(stack::LayerStack{FT}, frequency::Real,
                              source_layer::Int, obs_layer::Int,
                              k_rho_samples::Vector{FT};
                              z_source::FT=FT(NaN)) where {FT<:AbstractFloat}
    
    n_layers = stack.n_layers
    ω = 2π * frequency
    ε₀ = FT(8.854187817e-12)
    μ₀ = FT(4π * 1e-7)
    
    # Determine z_source position  
    # Default: midpoint of source layer (or small offset for half-spaces)
    s = source_layer
    if isnan(z_source)
        z_bot = stack.interfaces[s]
        z_top = stack.interfaces[s+1]
        if isinf(z_top)
            z_source = z_bot + FT(1e-6)  # small offset into half-space
        else
            z_source = (z_bot + z_top) / 2
        end
    end
    
    # Heights above/below within source layer
    h_below = z_source - stack.interfaces[s]      # distance to bottom of layer
    h_above = stack.interfaces[s+1] - z_source    # distance to top of layer
    d_layer = stack.layers[s].thickness             # total layer thickness
    
    n_samples = length(k_rho_samples)
    G_tilde_A = Vector{Complex{FT}}(undef, n_samples)
    G_tilde_phi = Vector{Complex{FT}}(undef, n_samples)
    
    for (idx, k_ρ) in enumerate(k_rho_samples)
        # ── Step 1: Vertical wavenumbers ──
        # k_{z,i} = √(k_i² − k_ρ²) with Im(k_z) ≤ 0 branch cut (proper Riemann sheet).
        k_z = compute_vertical_wavenumbers(stack, ω, k_ρ, ε₀, μ₀)
        
        # ── Step 2: Characteristic impedances ──
        # TE and TM modes have different impedances and thus different reflection
        # coefficients.  G̃_A is governed by TE reflections; G̃_φ by TM.
        Z_TE = compute_te_impedances(stack, k_z, ω, μ₀)
        Z_TM = compute_tm_impedances(stack, k_z, ω, ε₀)
        
        # ── Step 3: Two-direction TL recursion for TE ──
        # "Downward" recursion goes from the top half-space toward the ground and
        # yields Z_up[i] = input impedance looking UP from the bottom of layer i.
        # "Upward" recursion starts at the ground plane (Z_load = 0 for PEC) and
        # yields Z_down[i] = input impedance looking DOWN from the top of layer i.
        # The generalized reflection coefficients Γ_down[i], Γ_up[i] follow from
        # impedance mismatch at the source-layer boundaries.
        Z_up_TE = transmission_line_recursion_downward(Z_TE.Z, k_z, stack, n_layers)
        Z_down_TE = transmission_line_recursion_upward(Z_TE.Z, k_z, stack, n_layers)
        Γ_down_TE, Γ_up_TE = compute_generalized_reflection(Z_TE.Z, Z_down_TE, Z_up_TE, stack, n_layers)
        
        # ── Step 4: Two-direction TL recursion for TM (same structure) ──
        Z_up_TM = transmission_line_recursion_downward(Z_TM.Z, k_z, stack, n_layers)
        Z_down_TM = transmission_line_recursion_upward(Z_TM.Z, k_z, stack, n_layers)
        Γ_down_TM, Γ_up_TM = compute_generalized_reflection(Z_TM.Z, Z_down_TM, Z_up_TM, stack, n_layers)
        
        # ── Step 5: z-dependent spectral Green's function ──
        #
        # For source and observer in the SAME layer (source_layer == obs_layer = s),
        # the spectral GF with multiple reflections is (Michalski & Mosig 1997):
        #
        #   G̃(k_ρ, z, z) = ──────────1────────── × (1 + Γ↓ e^{-2jk_z h↓})
        #                    2 j k_{z,s}                                      
        #                                          × (1 + Γ↑ e^{-2jk_z h↑})
        #                                          ─────────────────────────
        #                                           1 − Γ↓ Γ↑ e^{-2jk_z d}
        #
        # where:
        #   h↓ = z − z_bottom  (source distance to bottom interface)
        #   h↑ = z_top − z     (source distance to top interface)
        #   d  = layer thickness
        #   Γ↓ = generalized reflection coeff looking toward ground
        #   Γ↑ = generalized reflection coeff looking toward free space
        #
        # The 1/(2jk_z) prefactor is the free-space spectral kernel.
        # The (1 + Γ↓ e^{…}) factors account for the reflected wave from each
        # boundary.  The denominator D handles multiple round-trip reflections.
        #
        # For a grounded slab (PEC at z=0), Γ↓ = −1 at the bottom of layer 1.
        # For the top half-space (layer N), Γ↑ = 0 (no reflection above).
        
        k_z_s = k_z[s]
        
        # Phase terms for z-dependence within the source layer
        exp_below_TE = exp(-2im * k_z_s * h_below)
        exp_above_TE = exp(-2im * k_z_s * h_above)
        
        # For an infinite (half-space) layer, h_above → ∞ ⟹ exp decay → 0
        if isinf(h_above)
            exp_above_TE = zero(Complex{FT})
        end
        
        # Multiple-reflection denominator: D = 1 − Γ↓ Γ↑ e^{-2jk_z d}
        # For a half-space (d=∞), D=1 because the round-trip phase vanishes.
        if isinf(d_layer)
            D_TE = one(Complex{FT})
            D_TM = one(Complex{FT})
        else
            exp_d = exp(-2im * k_z_s * d_layer)
            D_TE = 1 - Γ_down_TE[s] * Γ_up_TE[s] * exp_d
            D_TM = 1 - Γ_down_TM[s] * Γ_up_TM[s] * exp_d
        end
        
        # G̃_A: vector potential (TE only; Michalski-Mosig Form C, eq. 42)
        factor_down_TE = 1 + Γ_down_TE[s] * exp_below_TE
        factor_up_TE = 1 + Γ_up_TE[s] * exp_above_TE
        F_TE = factor_down_TE * factor_up_TE / D_TE
        G_tilde_A[idx] = (1 / (2im * k_z_s)) * F_TE
        
        # G̃_φ: scalar potential (TE/TM coupled; Michalski-Mosig Form C, eq. 50)
        # A horizontal current (RWG basis) radiates both TE and TM waves.
        # The scalar potential kernel is:  K̃^Φ = −jωε₀/k_ρ² × (V^h − V^e)
        # where V^h = (Z^TE/2)F^TE and V^e = (Z^TM/2)F^TM.
        # In the code convention (G = exp(−jkR)/R, no 4π):
        #   g̃_φ = −jωε₀/(2k_ρ²) × (Z^TE F^TE − Z^TM F^TM)
        # Reduces to 1/(2jk_z) in free space.  See verify_normalization.jl.
        factor_down_TM = 1 + Γ_down_TM[s] * exp_below_TE
        factor_up_TM = 1 + Γ_up_TM[s] * exp_above_TE
        F_TM = factor_down_TM * factor_up_TM / D_TM
        k_rho_sq = Complex{FT}(k_ρ)^2
        G_tilde_phi[idx] = (-im * ω * ε₀) / (2 * k_rho_sq) * (Z_TE.Z[s] * F_TE - Z_TM.Z[s] * F_TM)
    end
    
    return SpectralGF{FT}(k_rho_samples, G_tilde_A, G_tilde_phi, 
                          FT(frequency), source_layer, obs_layer)
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_vertical_wavenumbers(stack::LayerStack, ω, k_ρ, ε₀, μ₀)

Compute vertical wavenumber k_{z,i} for every layer at a given (real) k_ρ.

# Branch cut convention (CRITICAL)
We enforce `Im(k_z) ≤ 0` so that `e^{-jk_z z}` decays for z → +∞.
This selects the **proper Riemann sheet** under the `e^{+jωt}` time convention.

For finite-thickness slabs the TL recursion uses `tan(k_z h)` which is odd in
k_z, so the branch choice is immaterial.  It only matters for **half-space
layers** (layer 1 at the ground, layer N at the top) where the outgoing-wave
condition must be satisfied.

An earlier version enforced `Re(k_z) ≥ 0` instead — that is WRONG for
evanescent modes (k_ρ > k_i) where k_z is predominantly imaginary.
"""
function compute_vertical_wavenumbers(stack::LayerStack{FT}, ω::FT, k_ρ::FT,
                                       ε₀::FT, μ₀::FT) where {FT}
    n_layers = stack.n_layers
    k_z = Vector{Complex{FT}}(undef, n_layers)
    
    for i in 1:n_layers
        layer = stack.layers[i]
        μ_i = μ₀ * real(layer.mu_r)
        ε_i = ε₀ * effective_permittivity(layer, ω / (2π))
        
        # Layer wavenumber: k_i = ω √(μ_i ε_i)
        k_i = ω * sqrt(μ_i * ε_i)
        
        # k_{z,i} = √(k_i² − k_ρ²)
        # For propagating modes (k_ρ < k_i): k_z is real
        # For evanescent modes (k_ρ > k_i): k_z is imaginary
        k_z_sq = k_i^2 - k_ρ^2
        k_z[i] = sqrt(complex(k_z_sq))
        
        # Proper Riemann sheet: Im(k_z) ≤ 0
        if imag(k_z[i]) > 0
            k_z[i] = -k_z[i]
        end
    end
    
    return k_z
end

"""
    compute_te_impedances(stack::LayerStack, k_z, ω, μ₀)

Compute TE mode characteristic impedances for each layer.

# Formula
```
Z_i^TE = (ω μ_i) / k_{z,i}
```

Only the n_layers characteristic impedances are computed (no extra half-space slot).
"""
function compute_te_impedances(stack::LayerStack{FT}, k_z::Vector{Complex{FT}},
                                ω::FT, μ₀::FT) where {FT}
    n_layers = stack.n_layers
    Z = Vector{Complex{FT}}(undef, n_layers)
    
    for i in 1:n_layers
        μ_i = μ₀ * real(stack.layers[i].mu_r)
        Z[i] = (ω * μ_i) / k_z[i]
    end
    
    return TEImpedances{FT}(Z)
end

"""
    compute_tm_impedances(stack::LayerStack, k_z, ω, ε₀)

Compute TM mode characteristic impedances for each layer.

# Formula
```
Z_i^TM = k_{z,i} / (ω ε_i)
```
"""
function compute_tm_impedances(stack::LayerStack{FT}, k_z::Vector{Complex{FT}},
                                ω::FT, ε₀::FT) where {FT}
    n_layers = stack.n_layers
    Z = Vector{Complex{FT}}(undef, n_layers)
    
    for i in 1:n_layers
        ε_i = ε₀ * effective_permittivity(stack.layers[i], ω / (2π))
        Z[i] = k_z[i] / (ω * ε_i)
    end
    
    return TMImpedances{FT}(Z)
end

"""
    transmission_line_recursion_downward(Z, k_z, stack, n_layers)

Compute input impedances looking UPWARD by recursing from top → bottom.

Starting condition: Z_up[N] = Z_char[N]  (top half-space is matched).

At each layer, transforms the impedance through a TL section of length h_i
using the standard impedance transformation:

    Z_in = Z₀ (Z_L + jZ₀ tan θ) / (Z₀ + jZ_L tan θ),    θ = k_z h

Result: Z_up[i] = input impedance looking upward from the bottom of layer i.
Used to compute the upward-looking reflection coefficient Γ↑[i].
"""
function transmission_line_recursion_downward(Z::Vector{Complex{FT}}, 
                                              k_z::Vector{Complex{FT}},
                                              stack::LayerStack{FT}, 
                                              n_layers::Int) where {FT}
    Z_up = Vector{Complex{FT}}(undef, n_layers)
    
    # Top layer (half-space, infinite thickness): impedance = characteristic
    Z_up[n_layers] = Z[n_layers]
    
    # Recurse downward through finite-thickness layers
    for i in (n_layers - 1):-1:1
        h_i = stack.layers[i].thickness
        
        if isinf(h_i)
            Z_up[i] = Z[i]
        else
            θ_i = k_z[i] * h_i
            tan_θ = tan(θ_i)
            
            # Transform Z_up[i+1] (load above) through layer i
            numerator = Z_up[i+1] + im * Z[i] * tan_θ
            denominator = Z[i] + im * Z_up[i+1] * tan_θ
            
            Z_up[i] = Z[i] * numerator / denominator
        end
    end
    
    return Z_up
end

"""
    transmission_line_recursion_upward(Z, k_z, stack, n_layers)

Compute input impedances looking DOWNWARD by recursing from bottom → top.

Starting condition depends on ground plane:
- PEC ground (has_ground_plane=true):  Z_load = 0  (short circuit)
- No ground:                           Z_load = Z_char[1]  (matched)

The PEC short circuit is the critical boundary condition for microstrip
structures: an infinitely conducting ground plane forces the tangential
E-field to zero (Z=0), giving total reflection Γ = −1.

Result: Z_down[i] = input impedance looking downward from the top of layer i.
Used to compute the downward-looking reflection coefficient Γ↓[i].
"""
function transmission_line_recursion_upward(Z::Vector{Complex{FT}}, 
                                             k_z::Vector{Complex{FT}},
                                             stack::LayerStack{FT}, 
                                             n_layers::Int) where {FT}
    Z_down = Vector{Complex{FT}}(undef, n_layers)
    
    # ── Bottom boundary condition ──
    # PEC ground ⟹ Z = 0  (short circuit, Γ = −1)
    # No ground  ⟹ Z = Z_char[1]  (matched, Γ = 0)
    if stack.has_ground_plane
        Z_load_bottom = zero(Complex{FT})  # PEC short circuit
    else
        Z_load_bottom = Z[1]  # Open (match to first layer)
    end
    
    # First layer: transform from ground through layer 1
    h_1 = stack.layers[1].thickness
    if isinf(h_1)
        Z_down[1] = Z[1]
    else
        θ_1 = k_z[1] * h_1
        tan_θ = tan(θ_1)
        numerator = Z_load_bottom + im * Z[1] * tan_θ
        denominator = Z[1] + im * Z_load_bottom * tan_θ
        Z_down[1] = Z[1] * numerator / denominator
    end
    
    # Recurse upward: layer 2 to n_layers
    for i in 2:n_layers
        h_i = stack.layers[i].thickness
        
        if isinf(h_i)
            Z_down[i] = Z[i]
        else
            θ_i = k_z[i] * h_i
            tan_θ = tan(θ_i)
            
            # Load is Z_down[i-1] (impedance looking down from the top of layer i-1,
            # which is what layer i sees at its bottom boundary)
            numerator = Z_down[i-1] + im * Z[i] * tan_θ
            denominator = Z[i] + im * Z_down[i-1] * tan_θ
            
            Z_down[i] = Z[i] * numerator / denominator
        end
    end
    
    return Z_down
end

"""
    compute_generalized_reflection(Z, Z_down, Z_up, stack, n_layers)

Compute generalized (multi-layer) reflection coefficients for the source layer.

Returns `(Γ_down, Γ_up)` where:
- `Γ_down[i]` = reflection looking DOWN from inside layer i toward ground.
  For a PEC ground at the bottom of layer 1: Γ_down[1] = −1.
- `Γ_up[i]` = reflection looking UP from inside layer i toward free space.
  For the top half-space (layer N): Γ_up[N] = 0.

These are "generalized" because Z_down and Z_up already incorporate all
layers below/above (via the TL recursion), so the Γ values account for
the entire multi-layer geometry, not just a single interface.

The standard reflection coefficient formula is used:
    Γ = (Z_load − Z_char) / (Z_load + Z_char)
"""
function compute_generalized_reflection(Z::Vector{Complex{FT}}, 
                                         Z_down::Vector{Complex{FT}},
                                         Z_up::Vector{Complex{FT}},
                                         stack::LayerStack{FT},
                                         n_layers::Int) where {FT}
    Γ_down = Vector{Complex{FT}}(undef, n_layers)
    Γ_up = Vector{Complex{FT}}(undef, n_layers)
    
    for i in 1:n_layers
        # Reflection looking downward at bottom of layer i
        if i == 1
            if stack.has_ground_plane
                # PEC ground: Z_load = 0 → Γ = (0 - Z[i])/(0 + Z[i]) = -1
                Γ_down[i] = -one(Complex{FT})
            else
                # No ground plane, matched to layer 1 itself → no reflection
                Γ_down[i] = zero(Complex{FT})
            end
        else
            # Looking down from within layer i: sees Z_down[i-1] at interface
            Z_below = Z_down[i-1]
            Γ_down[i] = (Z_below - Z[i]) / (Z_below + Z[i])
        end
        
        # Reflection looking upward at top of layer i
        if i == n_layers
            # Top layer is half-space: no reflection above
            Γ_up[i] = zero(Complex{FT})
        else
            # Looking up from within layer i: sees Z_up[i+1] at interface
            Z_above = Z_up[i+1]
            Γ_up[i] = (Z_above - Z[i]) / (Z_above + Z[i])
        end
    end
    
    return Γ_down, Γ_up
end

# =============================================================================
# Sampling Functions
# =============================================================================

"""
    sample_real_axis(k0::FT, k_max::FT, n_samples::Int) where {FT}

Sample along real axis [0, k_max].

For Level 1 DCIM sampling (near-field).
"""
function sample_real_axis(k0::FT, k_max::FT, n_samples::Int) where {FT}
    return range(zero(FT), stop=k_max, length=n_samples)
end

"""
    sample_sommerfeld_branch_cut(k0::FT, T1::FT, n_samples::Int) where {FT}

Sample along Sommerfeld branch cut in complex k_ρ plane.

Path: k_ρ = k₀ + jt, where t ∈ [0, T₁]

For Level 2 DCIM sampling (far-field).
"""
function sample_sommerfeld_branch_cut(k0::FT, T1::FT, n_samples::Int) where {FT}
    # Path: k_ρ = k₀ + jt, t ∈ [0, T₁]
    # Return complex samples
    t = range(zero(FT), stop=T1, length=n_samples)
    return [k0 + im * ti for ti in t]
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    effective_wavenumber(layer::LayerInfo, ω, ε₀, μ₀)

Compute complex wavenumber for a layer.

# Formula
```
k = ω * sqrt(μ ε)
ε = ε₀ ε_r - j σ/ω   (complex permittivity)
μ = μ₀ μ_r
```
"""
function effective_wavenumber(layer::LayerInfo{FT}, ω::FT, ε₀::FT, μ₀::FT) where {FT}
    μ = μ₀ * real(layer.mu_r)
    ε = ε₀ * effective_permittivity(layer, ω / (2π))
    return ω * sqrt(μ * ε)
end

# =============================================================================
# Complex kρ Helper (for DCIM Level 2 path)
# =============================================================================

"""
    _spectral_gf_pair(stack, ω, source_layer, k_ρ, z_source, ε₀, μ₀) -> (G̃_A, G̃_φ)

Compute spectral GF at a single **complex** k_ρ value.

This is needed by the DCIM Level 2 fitting path, where k_ρ has both real and
imaginary parts (the path curves through the complex plane near the branch
point k_ρ = k₀).  The batch `compute_spectral_gf` only handles real k_ρ arrays.

Physics is identical to `compute_spectral_gf`: full TL recursion with TE/TM
separation, z-dependent formula, and PEC ground BC.  The only difference is
that k_z = √(k_i² − k_ρ²) is now complex even for lossless media.
"""
function _spectral_gf_pair(stack::LayerStack{FT}, ω::FT, source_layer::Int,
                            k_ρ::Complex{FT}, z_source::FT,
                            ε₀::FT, μ₀::FT) where {FT}
    n_layers = stack.n_layers
    s = source_layer

    # Vertical wavenumbers for all layers (complex kρ → complex kz)
    k_z = Vector{Complex{FT}}(undef, n_layers)
    Z_TE = Vector{Complex{FT}}(undef, n_layers)
    Z_TM = Vector{Complex{FT}}(undef, n_layers)
    for i in 1:n_layers
        layer = stack.layers[i]
        μ_i = μ₀ * real(layer.mu_r)
        ε_i = ε₀ * effective_permittivity(layer, ω / (2π))
        k_i_sq = ω^2 * μ_i * ε_i
        k_z_sq = complex(k_i_sq) - k_ρ^2
        k_z[i] = sqrt(k_z_sq)
        # Im(kz) ≤ 0: proper Riemann sheet (same convention as real-kρ version)
        if imag(k_z[i]) > 0
            k_z[i] = -k_z[i]
        end
        Z_TE[i] = (ω * μ_i) / k_z[i]
        Z_TM[i] = k_z[i] / (ω * ε_i)
    end

    # TL recursion + reflection (identical logic to the batch version)
    Z_up_TE = transmission_line_recursion_downward(Z_TE, k_z, stack, n_layers)
    Z_down_TE = transmission_line_recursion_upward(Z_TE, k_z, stack, n_layers)
    Γ_down_TE, Γ_up_TE = compute_generalized_reflection(Z_TE, Z_down_TE, Z_up_TE, stack, n_layers)

    Z_up_TM = transmission_line_recursion_downward(Z_TM, k_z, stack, n_layers)
    Z_down_TM = transmission_line_recursion_upward(Z_TM, k_z, stack, n_layers)
    Γ_down_TM, Γ_up_TM = compute_generalized_reflection(Z_TM, Z_down_TM, Z_up_TM, stack, n_layers)

    # z-dependent spectral GF (same formula as batch version)
    k_z_s = k_z[s]
    h_below = z_source - stack.interfaces[s]
    h_above = stack.interfaces[s+1] - z_source
    d_layer = stack.layers[s].thickness

    exp_below = exp(-2im * k_z_s * h_below)
    exp_above = isinf(h_above) ? zero(Complex{FT}) : exp(-2im * k_z_s * h_above)

    if isinf(d_layer)
        D_TE = one(Complex{FT})
        D_TM = one(Complex{FT})
    else
        exp_d = exp(-2im * k_z_s * d_layer)
        D_TE = 1 - Γ_down_TE[s] * Γ_up_TE[s] * exp_d
        D_TM = 1 - Γ_down_TM[s] * Γ_up_TM[s] * exp_d
    end

    # G̃_A: vector potential (TE only; Form C, eq. 42)
    factor_down_TE = 1 + Γ_down_TE[s] * exp_below
    factor_up_TE = 1 + Γ_up_TE[s] * exp_above
    F_TE = factor_down_TE * factor_up_TE / D_TE
    G_A = (1 / (2im * k_z_s)) * F_TE

    # G̃_φ: scalar potential (TE/TM coupled; Form C, eq. 50)
    # g̃_φ = −jωε₀/(2k_ρ²) × (Z^TE F^TE − Z^TM F^TM)
    factor_down_TM = 1 + Γ_down_TM[s] * exp_below
    factor_up_TM = 1 + Γ_up_TM[s] * exp_above
    F_TM = factor_down_TM * factor_up_TM / D_TM
    k_rho_sq = k_ρ^2  # k_ρ is already complex
    G_phi = (-im * ω * ε₀) / (2 * k_rho_sq) * (Z_TE[s] * F_TE - Z_TM[s] * F_TM)

    return (G_A, G_phi)
end

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, sg::SpectralGF{FT}) where {FT} = 
    print(io, "SpectralGF{$FT}($(length(sg.k_rho)) samples, " *
          "src=$(sg.source_layer), obs=$(sg.obs_layer))")

Base.summary(io::IO, sg::SpectralGF{FT}) where {FT} = 
    print(io, "Spectral GF with $(length(sg.k_rho)) samples " *
          "(layers $(sg.source_layer) → $(sg.obs_layer))")
