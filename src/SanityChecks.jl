## SanityChecks.jl
## Lightweight physical-law validation for MoM results.
##
## Design principles:
##   • Silent on pass (check_sparameter_quality prints one @info line only)
##   • Targeted @warn messages on failure with a likely-cause hint
##   • All checks are O(N²_ports) or O(N²_bf) — negligible vs. the matrix solve
##   • No side effects on the simulation data

# ─────────────────────────────────────────────────────────────────────────────
# 1. Impedance matrix symmetry  (EFIE only — MFIE/CFIE are not symmetric)
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_impedance_symmetry(Z; tol=1e-5, formulation=:EFIE) -> Bool

Verify `Z ≈ Zᵀ` (Frobenius relative norm).
Required for EFIE in passive, isotropic media; silently skipped for MFIE/CFIE.
Returns `true` when the check passes or is skipped.
"""
function check_impedance_symmetry(Z::AbstractMatrix;
                                  tol::Real         = 1e-5,
                                  formulation::Symbol = :EFIE)
    formulation !== :EFIE && return true
    normZ = norm(Z)
    normZ < eps(Float64) && return true           # degenerate matrix; skip
    err   = norm(Z .- transpose(Z)) / normZ
    err < tol && return true
    @warn """[MoM quality] Z-matrix not symmetric (EFIE)
  ‖Z − Zᵀ‖/‖Z‖ = $(round(err; sigdigits=3))  (tol = $tol)
  hint: non-reciprocal material, mesh asymmetry, or numerical accumulation"""
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. S-parameter reciprocity  (Sᵢⱼ = Sⱼᵢ)
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_sparameter_reciprocity(S; tol=1e-4) -> Bool

Verify `Sᵢⱼ = Sⱼᵢ` for all off-diagonal port pairs.
Required for passive, isotropic media without magneto-optic effects.
Returns `true` when the check passes.
"""
function check_sparameter_reciprocity(S::AbstractMatrix;
                                      tol::Real = 1e-4)
    n = size(S, 1)
    max_err = 0.0
    wi, wj  = 1, 2
    for i in 1:n
        for j in (i+1):n
            d = max(abs(S[i,j]), abs(S[j,i]), eps(Float64))
            e = abs(S[i,j] - S[j,i]) / d
            if e > max_err
                max_err = e
                wi, wj  = i, j
            end
        end
    end
    max_err < tol && return true
    @warn """[MoM quality] S-matrix not reciprocal
  max |Sᵢⱼ − Sⱼᵢ|/|Sᵢⱼ| = $(round(max_err; sigdigits=3))  at ($wi, $wj)  (tol = $tol)
  hint: solver under-converged, mesh asymmetry, or non-reciprocal medium"""
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Passivity  (I − SᴴS ⪰ 0)
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_passivity(S; tol=1e-6) -> Bool

Verify `I − SᴴS` is positive semi-definite (no power generation).
Returns `true` when the check passes.

The matrix `(I − SᴴS + (I − SᴴS)') / 2` is symmetrized before eigen-decomposition
to suppress rounding-induced imaginary parts.
"""
function check_passivity(S::AbstractMatrix;
                         tol::Real = 1e-6)
    n   = size(S, 1)
    SHS = S' * S
    P   = I(n) - SHS
    Psym = Hermitian((P .+ P') ./ 2)   # enforce Hermitian; removes O(ε_mach) asymmetry
    min_eig = minimum(real(eigvals(Psym)))
    min_eig >= -tol && return true
    @warn """[MoM quality] S-matrix not passive
  min λ(I − SᴴS) = $(round(min_eig; sigdigits=3))  (must be ≥ $(-tol))
  hint: numerical instability, non-physical material, or under-resolved mesh"""
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Combined S-parameter quality check
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_sparameter_quality(S; reciprocal_tol=1e-4, passivity_tol=1e-6)

Run reciprocity and passivity checks on an S-parameter matrix `S` and print a concise report.

- Pass: one `@info` line.
- Fail: one `@warn` block per violated check.

This function is called automatically by `computeSParameters` when `check_quality=true`
(the default).  It can also be called manually on any S-matrix.
"""
function check_sparameter_quality(S::AbstractMatrix;
                                  reciprocal_tol::Real = 1e-4,
                                  passivity_tol::Real  = 1e-6)
    r_ok = check_sparameter_reciprocity(S; tol = reciprocal_tol)
    p_ok = check_passivity(S; tol = passivity_tol)
    (r_ok && p_ok) && @info "[MoM quality] S-parameters OK  (reciprocity ✓  passivity ✓)"
    nothing
end
