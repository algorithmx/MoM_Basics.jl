# DCIM.jl
# Discrete Complex Image Method — Two-Level Aksun 1996 Algorithm
#
# Converts spectral domain Green's functions G̃(k_ρ) to spatial domain G(ρ)
# by approximating G̃ as a sum of complex exponentials in k_z, each of which
# maps to a spherical wave exp(−jkR)/R via the Sommerfeld identity — hence
# the name "complex images".
#
# Algorithm overview (Strata-verified):
#   1. Extract F(k_z) = k_z × G̃(k_z)  to remove the 1/k_z singularity.
#   2. Subtract the free-space asymptote F_∞ = −j/2 (quasi-static extraction).
#   3. Level 1: Sample F along a deep imaginary k_z path and fit with GPOF.
#      → Captures quasi-static / near-field behavior (dominant at small ρ).
#   4. Level 2: Sample F along a path near the branch point k_ρ = k₀
#      (where k_z₀ = 0), subtract Level 1's contribution, and fit the
#      residual with GPOF.
#      → Captures surface wave / guided wave behavior.
#   5. Convert GPOF exponentials to complex image amplitudes and distances,
#      multiply by the 2j convention factor (Sommerfeld identity + our
#      exp(−jkR)/R normalization).
#
# Convention: This code uses G(R) = exp(−jkR)/R  (NO 1/(4π) factor).
# The MoM kernel applies 1/(4π) externally.
#
# References:
# - M. I. Aksun, "A robust approach for the derivation of closed-form 
#   Green's functions," IEEE Trans. Microwave Theory Tech., vol. 44, 
#   pp. 651-658, 1996.
# - Verified against Strata C++ library (Sharma & Triverio, 2021,
#   https://github.com/modelics/strata)

# =============================================================================
# DCIM Coefficients Types (defined in LayerStack.jl, extended here)
# =============================================================================

"""
    DCIMFittingResult{FT<:AbstractFloat}

Result of DCIM fitting containing complex images for both levels.

Fields:
- `level1::Vector{ComplexImage{FT}}`: Deep imaginary path images (quasi-static tail)
- `level2::Vector{ComplexImage{FT}}`: Near-branch-point images (surface waves, near-field)
- `quasi_static::Vector{ComplexImage{FT}}`: Quasi-static images (currently unused)
- `sw_poles::Vector{Complex{FT}}`: Surface wave poles (currently unused)
- `n_samples_l1::Int`: Number of Level 1 samples
- `n_samples_l2::Int`: Number of Level 2 samples
"""
struct DCIMFittingResult{FT<:AbstractFloat}
    level1::Vector{ComplexImage{FT}}
    level2::Vector{ComplexImage{FT}}
    quasi_static::Vector{ComplexImage{FT}}
    sw_poles::Vector{Complex{FT}}
    n_samples_l1::Int
    n_samples_l2::Int
end

"""
    DCIMParameters{FT<:AbstractFloat}

Parameters for two-level DCIM fitting (Aksun 1996).

# Fields
- `T01::FT`: Level 1 (deep imaginary) t-range, dimensionless (default: 200)
- `T02_factor::FT`: Level 2 factor; T02 = T02_factor × √(max εr) (default: 5.0)
- `n_samples::Int`: Samples per level (default: 100)
- `svd_threshold::FT`: GPOF SVD truncation threshold (default: 1e-6)
- `max_images::Int`: Maximum complex images per level (default: 10)

# Strata defaults
T01 = 200, T02 = 5√εr_max, N = 100 per level.
"""
struct DCIMParameters{FT<:AbstractFloat}
    T01::FT
    T02_factor::FT
    n_samples::Int
    svd_threshold::FT
    max_images::Int
end

# Default constructor
function DCIMParameters{FT}(; T01::FT=FT(200.0),
                              T02_factor::FT=FT(5.0),
                              n_samples::Int=100,
                              svd_threshold::FT=FT(1e-6),
                              max_images::Int=10) where {FT<:AbstractFloat}
    DCIMParameters{FT}(T01, T02_factor, n_samples, svd_threshold, max_images)
end

# =============================================================================
# Main DCIM Algorithm
# =============================================================================

"""
    dcim_fit(stack, frequency, source_layer, obs_layer, k0;
             params, field_type, z_source) -> DCIMFittingResult

Two-level DCIM fitting (Aksun 1996, verified against Strata C++ library).

Paths in the complex kz plane (k = k₀ = free-space wavenumber):
- **Level 1** (deep imaginary, fitted FIRST): `kz = -jk(T02 + t)`, `t ∈ [0, T01]`
  Captures quasi-static tail. kρ = k√(1+(T02+t)²) is real and large.
  Starts where Level 2 ends; the two paths are contiguous.
- **Level 2** (near-branch-point, fitted SECOND): `kz = k(-jt + 1 - t/T02)`, `t ∈ [ε, T02]`
  Captures surface wave region. kρ is complex and sweeps through the
  branch point region kρ ≈ k₀ (where kz₀ = 0).
  At t≈0: kz ≈ k (real), kρ ≈ 0.  At t=T02: kz = -jkT02 (meets Level 1).

Level 1 is fitted first, then its contribution is subtracted from Level 2
spectral data before fitting Level 2 (sequential scheme, prevents double-counting).

Spectral extraction: F = kz × G̃ (Strata convention).
Convention: amplitudes include 2j factor so evaluate_dcim gives G = Σ a exp(-jkR)/R.
"""
function dcim_fit(stack::LayerStack{FT}, frequency::FT, 
                  source_layer::Int, obs_layer::Int, k0::FT;
                  params::DCIMParameters{FT}=DCIMParameters{FT}(), 
                  field_type::Symbol=:A,
                  z_source::FT=FT(NaN)) where {FT<:AbstractFloat}
    
    # ─── Parameters ───
    epsr_max = _max_relative_permittivity(stack, frequency)
    T02 = params.T02_factor * sqrt(epsr_max)
    T01 = params.T01
    N = params.n_samples
    k = k0  # Free-space wavenumber (= k_min for non-magnetic media)
    
    ω = 2π * frequency
    ε₀ = FT(8.854187817e-12)
    μ₀ = FT(4π * 1e-7)
    
    # Default z_source: midpoint of source layer
    if isnan(z_source)
        z_bot = stack.interfaces[source_layer]
        z_top = stack.interfaces[source_layer + 1]
        z_source = isinf(z_top) ? z_bot + FT(1e-6) : (z_bot + z_top) / 2
    end
    
    # ═══════════════════════════════════════════════════════════════════
    # Level 1: Deep imaginary kz path (fitted FIRST)
    # kz = -jk(T02 + t),  t ∈ [0, T01]
    # kρ = k√(1 + (T02+t)²)  [real — can use batch spectral GF]
    # This path starts where Level 2 ends and extends deeper.
    # ═══════════════════════════════════════════════════════════════════
    dt1 = T01 / FT(N - 1)
    t1 = FT[n * dt1 for n in 0:(N-1)]
    
    kz_L1 = [Complex{FT}(-im * k * (T02 + t)) for t in t1]
    krho_L1 = FT[k * sqrt(one(FT) + (T02 + t)^2) for t in t1]
    
    # Batch spectral GF at real kρ
    spec_L1 = compute_spectral_gf(stack, frequency, source_layer, obs_layer,
                                   krho_L1; z_source=z_source)
    G_L1 = field_type == :A ? spec_L1.G_tilde_A : spec_L1.G_tilde_phi
    
    # F = kz × G̃ (extracts 1/kz singularity — Strata convention)
    F_L1 = [kz_L1[n] * G_L1[n] for n in 1:N]
    
    # ─── Free-space extraction ───
    # The asymptotic F → F_∞ as kρ → ∞ depends on the field type:
    #   G̃_A:  F_A → −j/2   (always, independent of source layer)
    #   G̃_φ:  F_φ → −j/(2 εr_s)  (Form C: depends on source layer ε_r)
    # For G̃_φ in Form C, the spectral kernel is −jωε₀/(2k_ρ²)×(Z^TE F^TE − Z^TM F^TM).
    # As k_ρ→∞: Z^TE→0, Z^TM→k_z/(ωε_s), and k_ρ²≈−k_z², yielding F→−j/(2ε_{r,s}).
    # For a source in free space (ε_r=1), both reduce to −j/2.
    ε_r_source = FT(1)
    if field_type == :phi
        ε_r_source = abs(effective_permittivity(stack.layers[source_layer], frequency))
    end
    F_free = Complex{FT}(-im / (2 * ε_r_source))
    for n in 1:N
        F_L1[n] -= F_free
    end
    
    # ─── GPOF fit Level 1 ───
    F_max1 = maximum(abs.(F_L1))
    F_max1 = max(F_max1, eps(FT))
    
    # Raw Level 1 images: (α_kz, a_raw) in kz-space, BEFORE 2j convention factor
    L1_alpha = Complex{FT}[]
    L1_a_raw = Complex{FT}[]
    
    try
        gpof_L1 = gpof_fit(F_L1 / F_max1, dt1;
                           svd_threshold=params.svd_threshold,
                           max_order=params.max_images)
        
        for i in 1:gpof_L1.order
            s_i = gpof_L1.poles[i]
            R_i = gpof_L1.residues[i] * F_max1  # undo normalization
            
            # ── t → kz conversion for Level 1 ──
            # The GPOF fits: F(t) ≈ Σ R_i exp(s_i t),  t ∈ [0, T01]
            # The Level 1 path is: kz(t) = −jk(T02 + t)
            # Inverting:           t = jkz/k − T02
            #
            # Substituting into the exponential:
            #   exp(s_i t) = exp(s_i (j kz/k − T02))
            #              = exp(−s_i T02) × exp(s_i j kz/k)
            #
            # Since α_kz = s_i/(jk), we have s_i = α_kz × jk, and:
            #   s_i × j/k = α_kz × jk × j/k = α_kz × j² = −α_kz
            #
            # Therefore:
            #   exp(s_i j kz/k) = exp(−α_kz × kz)
            #
            # In kz-space the image is:
            #   a_raw    = R_i × exp(−s_i T02)
            #   F_L1(kz) = Σ a_raw × exp(−α_kz × kz)
            α_kz = s_i / (im * k)
            a_raw = R_i * exp(-T02 * s_i)
            
            push!(L1_alpha, α_kz)
            push!(L1_a_raw, a_raw)
        end
    catch e
        @warn "DCIM Level 1 GPOF failed: $e"
    end
    
    @info "DCIM Level 1: $(length(L1_a_raw)) images (T01=$T01, T02=$(round(T02;digits=2)), N=$N)"
    
    # ═══════════════════════════════════════════════════════════════════
    # Level 2: Near-branch-point path (complex kρ)
    # kz = k(-jt + 1 - t/T02),  t ∈ [ε, T02]
    # At t ≈ 0:   kz ≈ k (real),  kρ ≈ 0
    # At t ~ 1:   kz has O(k) imaginary part, kρ sweeps through the
    #             branch point region kρ ≈ k₀ (where kz₀ = 0)
    # At t = T02: kz = -jkT02 (joins Level 1 start)
    # ═══════════════════════════════════════════════════════════════════
    ε_t = FT(0.01)
    dt2 = (T02 - ε_t) / FT(N - 1)
    t2 = FT[ε_t + n * dt2 for n in 0:(N-1)]
    
    kz_L2 = [Complex{FT}(k * (-im * t + one(FT) - t / T02)) for t in t2]
    krho_L2 = [sqrt(Complex{FT}(k^2) - kz^2) for kz in kz_L2]
    
    # Ensure kρ in upper-right quadrant (Strata convention)
    for i in eachindex(krho_L2)
        krho_L2[i] = complex(abs(real(krho_L2[i])), abs(imag(krho_L2[i])))
    end
    
    # Compute spectral GF at complex kρ (one sample at a time via helper)
    F_L2 = Vector{Complex{FT}}(undef, N)
    for n in 1:N
        G_A_n, G_phi_n = _spectral_gf_pair(stack, ω, source_layer, krho_L2[n],
                                             z_source, ε₀, μ₀)
        G_n = field_type == :A ? G_A_n : G_phi_n
        F_L2[n] = kz_L2[n] * G_n
    end
    
    # Subtract free-space constant (same extraction as Level 1)
    for n in 1:N
        F_L2[n] -= F_free
    end
    
    # Check for NaN/Inf from spectral GF
    n_bad_pre = count(x -> !isfinite(abs(x)), F_L2)
    if n_bad_pre > 0
        @warn "DCIM Level 2: $n_bad_pre non-finite values in F_L2 before subtraction"
    end
    
    # ─── Subtract Level 1 contribution from Level 2 spectral data ───
    # The two-level scheme avoids double-counting by fitting Level 1 first,
    # then subtracting its spectral reconstruction from the Level 2 data.
    # In kz-space: F_L1(kz) = Σ a_raw_j exp(−α_j kz)
    # We evaluate this at each Level 2 kz sample point and subtract.
    for n in 1:N
        for j in eachindex(L1_a_raw)
            exponent = -L1_alpha[j] * kz_L2[n]
            if real(exponent) > 500  # Guard against overflow
                continue
            end
            F_L2[n] -= L1_a_raw[j] * exp(exponent)
        end
    end
    
    # Check for NaN/Inf after subtraction
    n_bad_post = count(x -> !isfinite(abs(x)), F_L2)
    if n_bad_post > 0
        @warn "DCIM Level 2: $n_bad_post non-finite values in F_L2 after subtraction"
        # Replace non-finite with zero to allow GPOF to proceed
        for n in 1:N
            if !isfinite(abs(F_L2[n]))
                F_L2[n] = zero(Complex{FT})
            end
        end
    end
    
    # ─── GPOF fit Level 2 (on corrected data) ───
    F_max2 = maximum(abs.(F_L2))
    F_max2 = max(F_max2, eps(FT))
    
    L2_alpha = Complex{FT}[]
    L2_a_raw = Complex{FT}[]
    
    try
        gpof_L2 = gpof_fit(F_L2 / F_max2, dt2;
                           svd_threshold=params.svd_threshold,
                           max_order=params.max_images)
        
        for i in 1:gpof_L2.order
            s_i = gpof_L2.poles[i]
            R_i = gpof_L2.residues[i] * F_max2  # undo normalization
            
            # ── t → kz conversion for Level 2 ──
            # The Level 2 path is: kz(t) = k(−jt + 1 − t/T02)
            # Inverting: t = T02(1 − kz/k) / (1 + jT02)
            #
            # The GPOF fits: F(t) ≈ Σ R_i exp(s_i t)
            # Substituting t(kz):
            #   exp(s_i t) = exp(s_i T02 (1 − kz/k) / (1 + jT02))
            #              = exp(s_i T02 / (1+jT02)) × exp(−[s_i T02 / ((1+jT02)k)] × kz)
            #
            # So: α_kz = s_i T02 / ((1 + jT02) k)
            #     a_raw = R_i × exp(k × α_kz)
            α_kz = s_i * T02 / ((one(FT) + im * T02) * k)
            kα = k * α_kz
            if real(kα) > 500
                @warn "DCIM Level 2 image $i: exp(k*α) overflow, Re(kα)=$(real(kα)), skipping"
                continue
            end
            a_raw = R_i * exp(kα)
            if !isfinite(abs(a_raw))
                @warn "DCIM Level 2 image $i: non-finite amplitude, skipping"
                continue
            end
            
            push!(L2_alpha, α_kz)
            push!(L2_a_raw, a_raw)
        end
    catch e
        @warn "DCIM Level 2 GPOF failed: $e"
    end
    
    @info "DCIM Level 2: $(length(L2_a_raw)) images"
    
    # ═══════════════════════════════════════════════════════════════════
    # Build final ComplexImage arrays with 2j convention factor
    #
    # The Sommerfeld identity maps spectral exponentials to spatial:
    #
    #   ∫₀^∞ [exp(−α kz) / kz] J₀(kρ ρ) kρ dkρ = j × exp(−jkR) / R
    #
    # where R = √(ρ² − α²)  (complex distance, via z_img = −jα so
    # R² = ρ² + z_img² = ρ² − α²).
    #
    # The spectral GF integral (our convention, no 1/(4π)) is:
    #
    #   G(ρ) = (1/2π) ∫ G̃(kρ) J₀(kρρ) kρ dkρ       [Eq. A]
    #
    # We extracted F = kz × G̃, so G̃ = F/kz, and:
    #
    #   G(ρ) = (1/2π) ∫ (F/kz) J₀ kρ dkρ
    #        ≈ (1/2π) ∫ (Σ a_raw exp(−α kz)) (1/kz) J₀ kρ dkρ
    #        = (1/2π) × j × Σ a_raw exp(−jkR)/R    [Sommerfeld identity]
    #
    # This gives the "standard" G with 1/(4π): for free space,
    # a_raw = −j/2, so G_std = j/(2π)×(−j/2)×exp/R = exp/(4πR).
    #
    # Our convention is G = exp(−jkR)/R (no 1/(4π)), i.e. G_ours = 4π × G_std.
    # Therefore: amplitude = 4π × j/(2π) × a_raw = 2j × a_raw.
    #
    # evaluate_dcim computes: G = Σ amplitude × exp(−jkR)/R
    # ═══════════════════════════════════════════════════════════════════
    level1_images = [ComplexImage{FT}(FT(2) * im * L1_a_raw[i], L1_alpha[i])
                     for i in eachindex(L1_a_raw)]
    level2_images = [ComplexImage{FT}(FT(2) * im * L2_a_raw[i], L2_alpha[i])
                     for i in eachindex(L2_a_raw)]
    
    # Quasi-static image from free-space extraction:
    # F_free = -j/(2 εr_s) maps to spatial domain with amplitude = 2j × F_free.
    #   G̃_A:  2j × (-j/2)     = 1                → exp(-jk₀R)/R
    #   G̃_φ:  2j × (-j/(2εr)) = 1/εr             → (1/εr) exp(-jk₀R)/R
    # distance = 0 so R = sqrt(ρ² + 0²) = ρ at z=z'.
    qs_amp = Complex{FT}(FT(2) * im * F_free)  # = 1/ε_r_source
    qs_images = [ComplexImage{FT}(qs_amp, zero(Complex{FT}))]
    
    return DCIMFittingResult{FT}(
        level1_images,
        level2_images,
        qs_images,           # quasi-static (free-space extraction image)
        Complex{FT}[],       # surface wave poles (captured by Level 2)
        N, N
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _max_relative_permittivity(stack, frequency) -> FT

Maximum |εr_eff| across all layers (used to set DCIM T02 parameter).
"""
function _max_relative_permittivity(stack::LayerStack{FT}, frequency::FT) where {FT}
    epsr_max = zero(FT)
    for layer in stack.layers
        epsr = abs(effective_permittivity(layer, frequency))
        if epsr > epsr_max
            epsr_max = epsr
        end
    end
    return epsr_max
end

# =============================================================================
# Spatial Evaluation
# =============================================================================

"""
    evaluate_dcim(fitting_result::DCIMFittingResult, ρ, z, z_prime, k)

Convenience wrapper: converts `DCIMFittingResult` to `DCIMCoefficients` and
delegates to the main `evaluate_dcim` below.
"""
function evaluate_dcim(fitting_result::DCIMFittingResult{FT}, ρ::FT, z::FT, z_prime::FT, k::Complex{FT}) where {FT}
    coeffs = DCIMCoefficients{FT}(
        fitting_result.quasi_static,
        fitting_result.level1,
        fitting_result.level2,
        fitting_result.sw_poles
    )
    return evaluate_dcim(coeffs, ρ, z, z_prime, k)
end

"""
    evaluate_dcim(coeffs::DCIMCoefficients, ρ, z, z_prime, k)

Evaluate the spatial Green's function from precomputed DCIM coefficients.

The Green's function is a sum over complex images:

    G(ρ, z, z') = Σ aᵢ exp(−jk Rᵢ) / Rᵢ

where the amplitude `aᵢ` includes the 2j convention factor (absorbed during
`dcim_fit`) and `Rᵢ` is a complex distance determined by the image type:

- **Quasi-static images**: R = √(ρ² + (z−z')² + α²)  [real α, physical image distance]
  These come from the free-space extraction (amplitude=1, distance=0 gives
  the direct free-space contribution exp(−jk₀R)/R).

- **Level 1 & 2 images**: R = √(ρ² + (z−z')² − α²)  [complex α from GPOF fitting]
  The minus sign is the Sommerfeld identity convention: each GPOF exponential
  in kz-space maps to exp(−jkR)/R in spatial domain with R² = ρ² + (z−z')² − α².

The wavenumber `k` is the **free-space** k₀ (not the substrate k), because
the DCIM fitting was performed in terms of k_z₀ (free-space vertical wavenumber).

# Arguments
- `coeffs::DCIMCoefficients`: Precomputed DCIM coefficients from `dcim_fit`
- `ρ::Real`: Horizontal distance between source and observer
- `z::Real`: z-coordinate of observer
- `z_prime::Real`: z-coordinate of source
- `k::Complex`: Free-space wavenumber k₀ = ω/c

# Notes
- The vertical separation (z − z') is essential for correct evaluation when
  source and observer are at different heights within the same layer.
- For source/observer in different layers, separate DCIM coefficients are
  needed for each layer pair (handled by `LayeredMediaGF`).
"""
function evaluate_dcim(coeffs::DCIMCoefficients{FT}, ρ::FT, z::FT, z_prime::FT, k::Complex{FT}) where {FT}
    result = zero(Complex{FT})
    k_real = real(k) > 0 ? real(k) : abs(k)
    
    # Vertical separation between source and observer
    dz = z - z_prime
    dz_sq = dz^2
    
    # Quasi-static images: R = √(ρ² + dz² + α²)  (plus sign — real image distance)
    for img in coeffs.quasi_static
        R = sqrt(complex(ρ^2 + dz_sq + img.distance^2))
        if abs(R) > eps(FT)
            result += img.amplitude * exp(-1im * k_real * R) / R
        end
    end
    
    # Level 1 + Level 2 complex images: R = √(ρ² + dz² − α²)  (minus sign — Sommerfeld)
    for imgs in (coeffs.level1, coeffs.level2)
        for img in imgs
            R = sqrt(complex(ρ^2 + dz_sq - img.distance^2))
            if abs(R) > eps(FT)
                result += img.amplitude * exp(-1im * k_real * R) / R
            end
        end
    end
    
    # Surface wave poles (placeholder — not yet extracted explicitly;
    # currently captured approximately by Level 2 GPOF fitting)
    for k_sw in coeffs.sw_poles
        result += exp(-im * k_sw * ρ) / sqrt(ρ + FT(1e-10))
    end
    
    return result
end

"""
    evaluate_dcim_smooth(coeffs::DCIMCoefficients, ρ, z, z_prime, k)

Evaluate only the smooth (non-singular) part of the DCIM expansion.

Excludes quasi-static images which contain the 1/R singularity (identical to
the free-space direct term already handled by singularity extraction).
Returns only Level 1 + Level 2 complex image contributions.

This is used for self-term (coincident triangle) integration where the
singularity is already handled by `greenfunc_star`, and only the smooth
correction from the layered media needs to be added.

# Arguments
- `coeffs::DCIMCoefficients`: Precomputed DCIM coefficients from `dcim_fit`
- `ρ::Real`: Horizontal distance between source and observer
- `z::Real`: z-coordinate of observer
- `z_prime::Real`: z-coordinate of source
- `k::Complex`: Free-space wavenumber k₀ = ω/c

# Notes
- Includes vertical separation (z − z') in the complex distance calculation.
- Returns only the smooth part (Level 1 & 2 complex images + surface waves).
"""
function evaluate_dcim_smooth(coeffs::DCIMCoefficients{FT}, ρ::FT, z::FT, z_prime::FT, k::Complex{FT}) where {FT}
    result = zero(Complex{FT})
    k_real = real(k) > 0 ? real(k) : abs(k)
    
    # Vertical separation between source and observer
    dz = z - z_prime
    dz_sq = dz^2

    # Only Level 1 + Level 2 complex images (smooth, no singularity)
    for imgs in (coeffs.level1, coeffs.level2)
        for img in imgs
            R = sqrt(complex(ρ^2 + dz_sq - img.distance^2))
            if abs(R) > eps(FT)
                result += img.amplitude * exp(-1im * k_real * R) / R
            end
        end
    end

    # Surface wave poles (if any — typically captured by Level 2)
    for k_sw in coeffs.sw_poles
        result += exp(-im * k_sw * ρ) / sqrt(ρ + FT(1e-10))
    end

    return result
end

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, result::DCIMFittingResult{FT}) where {FT} = 
    print(io, "DCIMFittingResult{$FT}(L1=$(length(result.level1)), " *
          "L2=$(length(result.level2)), qs=$(length(result.quasi_static)))")

Base.summary(io::IO, result::DCIMFittingResult{FT}) where {FT} = 
    print(io, "DCIM fit: $(length(result.level1)) + $(length(result.level2)) images, " *
          "$(length(result.sw_poles)) surface wave poles")
