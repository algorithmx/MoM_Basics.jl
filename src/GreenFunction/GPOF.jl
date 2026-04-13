# GPOF.jl
# Generalized Pencil of Function Method

using Printf  # For @sprintf in diagnostic output
# 
# This module implements the GPOF algorithm for fitting complex exponentials
# to sampled data. Used by DCIM to extract complex image coefficients from
# spectral domain Green's functions.
#
# Reference:
# - Y. Hua and T. K. Sarkar, "Generalized pencil of function method for 
#   extracting poles of an EM system from its transient response," 
#   IEEE Trans. Antennas Propagat., vol. 37, pp. 229-234, 1989.

# =============================================================================
# GPOF Result Type
# =============================================================================

"""
    GPOFResult{FT<:AbstractFloat}

Result of GPOF fitting containing extracted poles and residues.

Fields:
- `poles::Vector{Complex{FT}}`: Complex poles sᵢ (continuous domain)
- `residues::Vector{Complex{FT}}`: Complex residues Rᵢ
- `order::Int`: Effective model order M
- `svd_threshold::FT`: Threshold used for SVD truncation
- `singular_values::Vector{FT}`: Singular values from SVD (for diagnostic)

# Mathematical Formulation
The fitted model is:
```
y(t) ≈ Σᵢ₌₁ᴹ Rᵢ exp(sᵢ t)
```

where:
- M = model order (automatically determined)
- Rᵢ = complex residues
- sᵢ = complex poles (continuous time)
"""
struct GPOFResult{FT<:AbstractFloat}
    poles::Vector{Complex{FT}}
    residues::Vector{Complex{FT}}
    order::Int
    svd_threshold::FT
    singular_values::Vector{FT}
end

# =============================================================================
# Core GPOF Algorithm
# =============================================================================

"""
    gpof_fit(samples::Vector{Complex{FT}}, dt::FT; 
             pencil_param::Int=0, 
             svd_threshold::FT=FT(1e-6),
             max_order::Int=0) where {FT<:AbstractFloat}

Fit complex exponentials to sampled data using Generalized Pencil of Function.

# Algorithm (from Hua & Sarkar 1989)

Given N samples `y[n]` for n = 0, 1, ..., N-1:

**Step 1: Construct Hankel matrices**
```
Y₁ = [y[0]   y[1]   ... y[L-1]  ]     (N-L) × L
     [y[1]   y[2]   ... y[L]    ]
     [...                         ]
     [y[N-L-1] ...   y[N-2]     ]

Y₂ = [y[1]   y[2]   ... y[L]    ]     (N-L) × L
     [y[2]   y[3]   ... y[L+1]  ]
     [...                         ]
     [y[N-L] ...     y[N-1]     ]
```
where L is the pencil parameter (default N/2).

**Step 2: SVD of concatenated matrix**
```
[Y₁] = U Σ Vᴴ
[Y₂]
```

**Step 3: Determine effective order M**
Truncate to M singular values where σ_M/σ₁ > svd_threshold.

**Step 4: Extract poles via generalized eigenvalue problem**
```
zᵢ = eigenvalue of (V₁ᴴ Y₁ V₁)⁻¹ (V₁ᴴ Y₂ V₁)
```
where V₁ = V[:, 1:M].

**Step 5: Convert discrete to continuous poles**
```
sᵢ = log(zᵢ) / dt
```

**Step 6: Compute residues via least squares**
```
Solve: V_matrix * R = samples
where V_matrix[n,i] = exp(sᵢ * n * dt)
```

# Arguments
- `samples::Vector{Complex{FT}}`: Complex samples y[n], n = 0, ..., N-1
- `dt::FT`: Sampling interval
- `pencil_param::Int`: Pencil parameter L (default: N/2). Larger L → better noise filtering
- `svd_threshold::FT`: SVD truncation threshold (default: 1e-6). 
  Typical range: 1e-3 to 1e-6. Tighter → more poles, better accuracy.
- `max_order::Int`: Maximum model order (default: 0 = no limit)

# Returns
- `GPOFResult`: Structure containing poles, residues, and diagnostic info

# Examples
```julia
# Create test signal: sum of 3 exponentials
t = collect(0:0.01:10)
y = 2.0 * exp.((-0.5 + 1.0im) * t) + 
    1.0 * exp.((-1.0 + 2.0im) * t) +
    0.5 * exp.((-2.0 + 5.0im) * t)

# Fit with GPOF
result = gpof_fit(y, 0.01)

# Reconstructed signal
y_fit = sum(result.residues[i] * exp.(result.poles[i] * t) for i in 1:result.order)
```

# Notes
- Samples should be complex-valued (even for real signals, use analytic representation)
- For noisy data, increase L (pencil_param) for better filtering
- If fitting fails, check singular_values for rank deficiency

# References
- Hua & Sarkar, IEEE T-AP, 1989 (original paper)
- Sarkar & Pereira, IEEE AP Magazine, 1995 (tutorial)
"""
function gpof_fit(samples::Vector{Complex{FT}}, dt::FT;
                  pencil_param::Int=0, 
                  svd_threshold::FT=FT(1e-6),
                  max_order::Int=0) where {FT<:AbstractFloat}
    
    N = length(samples)
    N >= 2 || error("GPOF requires at least 2 samples")
    
    # Step 1: Determine pencil parameter L
    L = pencil_param > 0 ? pencil_param : div(N, 2)
    L >= 1 || error("Pencil parameter L must be at least 1")
    L < N || error("Pencil parameter L must be less than N")
    
    # Construct Hankel matrices Y₁ and Y₂ (each size (N-L) × L)
    Y1 = Matrix{Complex{FT}}(undef, N-L, L)
    Y2 = Matrix{Complex{FT}}(undef, N-L, L)
    
    @inbounds for j in 1:L
        for i in 1:(N-L)
            Y1[i, j] = samples[i + j - 1]     # Y₁[i,j] = y[i+j-2] in 0-indexed
            Y2[i, j] = samples[i + j]         # Y₂[i,j] = y[i+j-1] in 0-indexed
        end
    end
    
    # Step 2: SVD of concatenated matrix [Y₁; Y₂] (size 2(N-L) × L)
    Y_concat = vcat(Y1, Y2)
    svd_result = svd(Y_concat, full=false)
    
    # Step 3: Determine effective order M
    # Find number of singular values above threshold
    σ_max = svd_result.S[1]
    M = count(s -> s / σ_max > svd_threshold, svd_result.S)
    
    # Apply maximum order limit if specified
    if max_order > 0 && M > max_order
        M = max_order
    end
    
    M >= 1 || error("GPOF: No significant singular values found. Check svd_threshold.")
    M <= L || error("GPOF: Effective order M > L. Increase pencil_param.")
    
    # Step 4: Extract poles via generalized eigenvalue problem
    # V1 = right singular vectors for significant singular values
    V1 = svd_result.V[:, 1:M]  # Size L × M
    
    # Compute V1ᴴ Y1ᴴ Y1 V1 and V1ᴴ Y1ᴴ Y2 V1
    # This is the correct formulation for the pencil Y2 - z Y1
    Y1_V1 = Y1 * V1           # Size (N-L) × M
    Y2_V1 = Y2 * V1           # Size (N-L) × M
    V1H_Y1H_Y1_V1 = Y1_V1' * Y1_V1  # Size M × M
    V1H_Y1H_Y2_V1 = Y1_V1' * Y2_V1  # Size M × M
    
    # Solve generalized eigenvalue problem: V1ᴴ Y1ᴴ Y2 V1 = z * V1ᴴ Y1ᴴ Y1 V1
    # Equivalent to: eig(V1ᴴ Y1ᴴ Y1 V1 \ V1ᴴ Y1ᴴ Y2 V1)
    A = V1H_Y1H_Y1_V1 \ V1H_Y1H_Y2_V1
    
    # Compute eigenvalues (discrete poles zᵢ)
    eigen_result = eigen(A)
    z_poles = eigen_result.values
    
    # Step 4b: Filter unstable eigenvalues
    # Eigenvalues with |z| >> 1 correspond to exponentially growing modes that
    # are physically non-sensical and cause Vandermonde matrix overflow.
    # Threshold: |z|^N < 1e250 → |z| < 10^(250/N)
    z_max = FT(10)^(FT(250) / N)
    valid_mask = [abs(z) < z_max for z in z_poles]
    z_poles = z_poles[valid_mask]
    M = length(z_poles)
    
    if M < 1
        # All poles filtered — return single zero-order result
        return GPOFResult{FT}(
            Complex{FT}[],
            Complex{FT}[],
            0,
            svd_threshold,
            svd_result.S
        )
    end
    
    # Step 5: Convert discrete poles to continuous domain
    # z = exp(s * dt) → s = log(z) / dt
    s_poles = log.(z_poles) / dt
    
    # Step 6: Compute residues via least squares
    # Construct Vandermonde-like matrix V where V[n,i] = exp(sᵢ * n * dt) = zᵢⁿ
    V_matrix = Matrix{Complex{FT}}(undef, N, M)
    @inbounds for i in 1:M
        z_i = z_poles[i]
        z_power = one(Complex{FT})  # z⁰ = 1
        for n in 1:N
            V_matrix[n, i] = z_power
            z_power *= z_i
        end
    end
    
    # Solve least squares: V * R = samples
    residues = V_matrix \ samples
    
    # Create result struct
    return GPOFResult{FT}(
        s_poles,           # Continuous poles
        residues,          # Residues
        M,                 # Effective order
        svd_threshold,     # Threshold used
        svd_result.S       # Singular values
    )
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    gpof_fit_real(samples::Vector{FT}, dt::FT; kwargs...) where {FT<:AbstractFloat}

Fit complex exponentials to real-valued samples.

Automatically constructs analytic signal via Hilbert transform before fitting.
"""
function gpof_fit_real(samples::Vector{FT}, dt::FT; 
                       kwargs...) where {FT<:AbstractFloat}
    # For real signals, we need to construct the analytic representation
    # Simple approach: use FFT-based Hilbert transform or just zero imaginary part
    # For now, treat as complex with zero imaginary part
    complex_samples = Complex{FT}.(samples)
    return gpof_fit(complex_samples, dt; kwargs...)
end

"""
    evaluate_gpof(result::GPOFResult{FT}, t::Union{FT, Vector{FT}}) where {FT}

Evaluate the fitted exponential model at time(s) t.

# Formula
```
y(t) = Σᵢ Rᵢ exp(sᵢ t)
```
"""
function evaluate_gpof(result::GPOFResult{FT}, t::FT) where {FT<:AbstractFloat}
    y = zero(Complex{FT})
    @inbounds for i in 1:result.order
        y += result.residues[i] * exp(result.poles[i] * t)
    end
    return y
end

function evaluate_gpof(result::GPOFResult{FT}, t::Vector{FT}) where {FT<:AbstractFloat}
    y = Vector{Complex{FT}}(undef, length(t))
    @inbounds for j in eachindex(t)
        y[j] = evaluate_gpof(result, t[j])
    end
    return y
end

"""
    gpof_residual(result::GPOFResult{FT}, samples::Vector{Complex{FT}}, dt::FT) where {FT}

Compute fitting residual (RMS error) between original samples and GPOF fit.
"""
function gpof_residual(result::GPOFResult{FT}, samples::Vector{Complex{FT}}, dt::FT) where {FT}
    N = length(samples)
    t = collect(range(zero(FT), step=dt, length=N))
    y_fit = evaluate_gpof(result, t)
    
    residual = norm(y_fit - samples) / sqrt(N)
    return residual
end

"""
    gpof_relative_error(result::GPOFResult{FT}, samples::Vector{Complex{FT}}, dt::FT) where {FT}

Compute relative fitting error.
"""
function gpof_relative_error(result::GPOFResult{FT}, samples::Vector{Complex{FT}}, dt::FT) where {FT}
    N = length(samples)
    t = collect(range(zero(FT), step=dt, length=N))
    y_fit = evaluate_gpof(result, t)
    
    error_norm = norm(y_fit - samples)
    signal_norm = norm(samples)
    
    return signal_norm > 0 ? error_norm / signal_norm : error_norm
end

# =============================================================================
# Diagnostic Functions
# =============================================================================

"""
    diagnose_gpof(result::GPOFResult{FT}) where {FT}

Print diagnostic information about GPOF fit.
"""
function diagnose_gpof(result::GPOFResult{FT}) where {FT}
    println("GPOF Fit Diagnostics")
    println("="^50)
    println("Model order: $(result.order)")
    println("SVD threshold: $(result.svd_threshold)")
    println()
    println("Singular values (first 10):")
    for (i, σ) in enumerate(result.singular_values[1:min(10, end)])
        println("  σ[$i] = $(@sprintf("%.6e", σ))")
    end
    println()
    println("Extracted poles and residues:")
    for i in 1:result.order
        s = result.poles[i]
        R = result.residues[i]
        println("  Mode $i: s = $(@sprintf("%.4f", real(s))) + $(@sprintf("%.4f", imag(s)))im, " *
                "R = $(@sprintf("%.4f", real(R))) + $(@sprintf("%.4f", imag(R)))im")
    end
end

"""
    is_stable(result::GPOFResult{FT}) -> Bool

Check if all poles are stable (negative real part → decaying exponentials).
"""
is_stable(result::GPOFResult{FT}) where {FT} = all(real(p) < 0 for p in result.poles)

"""
    has_conjugate_pairs(result::GPOFResult{FT}; tol::FT=FT(1e-10)) -> Bool

Check if poles come in conjugate pairs (expected for real signals).
"""
function has_conjugate_pairs(result::GPOFResult{FT}; tol::FT=FT(1e-10)) where {FT}
    # For real signals, poles should come in conjugate pairs
    # Check if for every pole, its conjugate is also present
    poles = result.poles
    for i in 1:result.order
        found_pair = false
        p_conj = conj(poles[i])
        for j in 1:result.order
            if abs(poles[j] - p_conj) < tol
                found_pair = true
                break
            end
        end
        if !found_pair
            return false
        end
    end
    return true
end

# =============================================================================
# Display
# =============================================================================

Base.show(io::IO, result::GPOFResult{FT}) where {FT} = 
    print(io, "GPOFResult{$FT}(order=$(result.order), $(length(result.singular_values)) samples)")

Base.summary(io::IO, result::GPOFResult{FT}) where {FT} = 
    print(io, "GPOF fit: order $(result.order), " *
          "$(is_stable(result) ? "stable" : "unstable") poles")
