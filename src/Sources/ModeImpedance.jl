"""
Waveguide mode impedance calculations for S-parameter extraction.

This module provides functions to compute cutoff frequencies and
characteristic impedances for TE and TM modes in rectangular waveguides.
"""

"""
    compute_mode_cutoff(m::Int, n::Int, a::FT, b::FT,
                        [epsilon::FT, mu::FT]) where {FT<:Real} -> FT

Compute the cutoff frequency of a rectangular waveguide mode.

The cutoff frequency is given by:
```
f_c = (c/2) × √((m/a)² + (n/b)²)
```

where `c = 1/√(με)` is the speed of light in the medium.

# Arguments
- `m::Int` -- Mode index along width (a) direction
- `n::Int` -- Mode index along height (b) direction
- `a::FT` -- Waveguide width (broad dimension)
- `b::FT` -- Waveguide height (narrow dimension)
- `epsilon::FT` -- Permittivity (default: ε₀)
- `mu::FT` -- Permeability (default: μ₀)

# Returns
- Cutoff frequency in Hz

# Example
```julia
# TE₁₀ cutoff for WR-90 (a=22.86mm, b=10.16mm)
f_c = compute_mode_cutoff(1, 0, 0.02286, 0.01016)  # ≈ 6.56 GHz
```
"""
function compute_mode_cutoff(
    m::Int, n::Int, a::FT, b::FT,
    epsilon::FT=FT(ε_0), mu::FT=FT(μ_0)
) where {FT<:Real}
    c = 1 / sqrt(mu * epsilon)
    return (c / 2) * sqrt((m/a)^2 + (n/b)^2)
end

"""
    compute_mode_impedance(mode::Symbol, m::Int, n::Int, f::FT, a::FT, b::FT;
                           [epsilon::FT, mu::FT]) where {FT<:Real} -> FT

Compute the characteristic impedance of a rectangular waveguide mode.

For propagating modes (f > f_c):
- TE modes: Z_TE = η / √(1 - (f_c/f)²)
- TM modes: Z_TM = η × √(1 - (f_c/f)²)

where η = √(μ/ε) is the intrinsic impedance of the medium.

For evanescent modes (f ≤ f_c), returns `Inf`.

# Arguments
- `mode::Symbol` -- Mode type: `:TE` or `:TM`
- `m::Int` -- Mode index along width (a) direction
- `n::Int` -- Mode index along height (b) direction
- `f::FT` -- Operating frequency in Hz
- `a::FT` -- Waveguide width (broad dimension)
- `b::FT` -- Waveguide height (narrow dimension)
- `epsilon::FT` -- Permittivity (default: ε₀)
- `mu::FT` -- Permeability (default: μ₀)

# Returns
- Mode impedance in Ohms (or `Inf` for evanescent modes)

# Example
```julia
# TE₁₀ impedance for WR-90 at 10 GHz
Z_te10 = compute_mode_impedance(:TE, 1, 0, 10e9, 0.02286, 0.01016)  # ≈ 465 Ω
```
"""
function compute_mode_impedance(
    mode::Symbol, m::Int, n::Int, f::FT, a::FT, b::FT;
    epsilon::FT=FT(ε_0), mu::FT=FT(μ_0)
) where {FT<:Real}
    mode in (:TE, :TM) || error("mode must be :TE or :TM, got :$mode")

    f_c = compute_mode_cutoff(m, n, a, b, epsilon, mu)

    if f <= f_c
        return FT(Inf)  # Evanescent mode
    end

    eta = sqrt(mu / epsilon)
    factor = sqrt(1 - (f_c/f)^2)

    return mode == :TE ? eta / factor : eta * factor
end


