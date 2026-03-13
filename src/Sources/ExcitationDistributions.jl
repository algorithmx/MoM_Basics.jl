"""
    AbstractExcitationDistribution{FT<:Real}

Abstract type hierarchy for port excitation distributions.

Subtypes define how voltage is distributed across port edges.
"""
abstract type AbstractExcitationDistribution{FT<:Real} end

# -----------------------------------------------------------------------------
# UniformDistribution - constant weight
# -----------------------------------------------------------------------------

"""
    UniformDistribution{FT<:Real} <: AbstractExcitationDistribution{FT}

Uniform excitation distribution across all port edges.

All edges receive equal voltage weight, useful for testing or special cases.
"""
struct UniformDistribution{FT<:Real} <: AbstractExcitationDistribution{FT} end

"""
    UniformDistribution(; FT::Type{<:Real}=Float64)

Construct a uniform excitation distribution.
"""
UniformDistribution(; FT::Type{<:Real}=Float64) = UniformDistribution{FT}()

# -----------------------------------------------------------------------------
# SingleSideDistribution - excite one side of rectangular port
# -----------------------------------------------------------------------------

"""
    SingleSideDistribution{FT<:Real} <: AbstractExcitationDistribution{FT}

Excitation distribution for a single side of a rectangular port.

Applies uniform voltage to edges on one side while keeping the other three sides at zero.
Useful for exciting specific port edges or testing individual boundary contributions.

# Fields
- `side::Symbol` -- Which side to excite: `:left`, `:right`, `:bottom`, or `:top`
- `tol::FT` -- Tolerance for determining if an edge is on the specified side

# Coordinate System
For a rectangular port centered at origin:
- `:left`   -- x = -width/2  (xi = 0)
- `:right`  -- x = +width/2  (xi = 1)
- `:bottom` -- y = -height/2 (eta = 0)
- `:top`    -- y = +height/2 (eta = 1)

# Example
```julia
# Excite only the left side of the port
left_side = SingleSideDistribution(:left)

# Excite bottom side with custom tolerance
bottom = SingleSideDistribution(:bottom, tol=1e-4)
```
"""
struct SingleSideDistribution{FT<:Real} <: AbstractExcitationDistribution{FT}
    side::Symbol
    tol::FT
    
    function SingleSideDistribution{FT}(side::Symbol, tol::FT=FT(1e-6)) where {FT<:Real}
        side in (:left, :right, :bottom, :top) || 
            error("side must be :left, :right, :bottom, or :top, got :$side")
        tol > 0 || error("tolerance must be positive, got $tol")
        return new{FT}(side, tol)
    end
end

"""
    SingleSideDistribution(side::Symbol; tol::Real=1e-6, FT::Type{<:Real}=Float64)

Construct a single-side excitation distribution.

# Arguments
- `side::Symbol` -- Side to excite: `:left`, `:right`, `:bottom`, or `:top`
- `tol::Real` -- Tolerance for edge detection (default: 1e-6)
- `FT::Type{<:Real}` -- Float precision type (default: Float64)
"""
function SingleSideDistribution(side::Symbol; tol::Real=1e-6, FT::Type{<:Real}=Float64)
    return SingleSideDistribution{FT}(side, FT(tol))
end

# Convenience constructors for each side
"""
    LeftSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64)

Excite the left side of the rectangular port (x = -width/2).
"""
LeftSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64) = SingleSideDistribution{FT}(:left, FT(tol))

"""
    RightSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64)

Excite the right side of the rectangular port (x = +width/2).
"""
RightSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64) = SingleSideDistribution{FT}(:right, FT(tol))

"""
    BottomSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64)

Excite the bottom side of the rectangular port (y = -height/2).
"""
BottomSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64) = SingleSideDistribution{FT}(:bottom, FT(tol))

"""
    TopSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64)

Excite the top side of the rectangular port (y = +height/2).
"""
TopSideDistribution(; tol::Real=1e-6, FT::Type{<:Real}=Float64) = SingleSideDistribution{FT}(:top, FT(tol))

# -----------------------------------------------------------------------------
# CustomDistribution - user-defined function
# -----------------------------------------------------------------------------

"""
    CustomDistribution{FT<:Real, F<:Function} <: AbstractExcitationDistribution{FT}

Custom excitation distribution defined by a user-provided function.

# Fields
- `weight_func::F` -- Function `(ξ, η) -> Complex{FT}` where ξ, η ∈ [0, 1]
  are normalized coordinates (ξ along width, η along height)

# Example
```julia
# Gaussian distribution centered at port center
custom = CustomDistribution((ξ, η) -> exp(-((ξ-0.5)^2 + (η-0.5)^2) * 10))
```
"""
struct CustomDistribution{FT<:Real, F<:Function} <: AbstractExcitationDistribution{FT}
    weight_func::F
end

"""
    CustomDistribution(func::Function; FT::Type{<:Real}=Float64)

Construct a custom distribution from a function.

The function should take normalized coordinates `(ξ, η)` where both are in [0, 1],
with (0,0) at the lower-left corner and (1,1) at the upper-right corner.
"""
function CustomDistribution(func::Function; FT::Type{<:Real}=Float64)
    return CustomDistribution{FT, typeof(func)}(func)
end

# -----------------------------------------------------------------------------
# Polymorphic voltage computation
# -----------------------------------------------------------------------------

"""
    compute_voltage(
        dist::AbstractExcitationDistribution{FT},
        edge_center::MVec3D{FT},
        port_params::NamedTuple
    ) -> Complex{FT}

Compute the complex voltage weight for an edge based on the excitation distribution.

# Arguments
- `dist` -- Excitation distribution
- `edge_center` -- 3D position of the edge center
- `port_params` -- NamedTuple with port geometry:
  - `center` -- Port center position
  - `widthDir` -- Unit vector along port width
  - `heightDir` -- Unit vector along port height
  - `normal` -- Port normal vector
  - `width` -- Port width
  - `height` -- Port height

# Returns
- Complex voltage weight for the edge
"""
function compute_voltage end

"""
    compute_voltage(dist::UniformDistribution{FT}, args...) -> Complex{FT}

Compute voltage weight for uniform distribution (always returns 1.0).
"""
compute_voltage(dist::UniformDistribution{FT}, args...) where {FT<:Real} = Complex{FT}(1.0)

"""
    compute_voltage(dist::SingleSideDistribution{FT}, edge_center, port_params) -> Complex{FT}

Compute voltage weight for single-side excitation.

Returns 1.0 if the edge center is on the specified side of the rectangle,
0.0 otherwise. The side is determined by checking if the normalized coordinate
is within tolerance of the boundary.

# Side detection
- `:left`   -- ξ ≈ 0 (xi < tol)
- `:right`  -- ξ ≈ 1 (xi > 1 - tol)
- `:bottom` -- η ≈ 0 (eta < tol)
- `:top`    -- η ≈ 1 (eta > 1 - tol)
"""
function compute_voltage(
    dist::SingleSideDistribution{FT},
    edge_center::MVec3D{FT},
    port_params::NamedTuple
) where {FT<:Real}
    u, v, _ = _rectangular_port_local_coordinates(
        edge_center,
        port_params.center,
        port_params.widthDir,
        port_params.heightDir,
        port_params.normal
    )
    xi = u / port_params.width + FT(0.5)
    eta = v / port_params.height + FT(0.5)
    
    tol = dist.tol
    on_side = if dist.side == :left
        xi < tol
    elseif dist.side == :right
        xi > (1 - tol)
    elseif dist.side == :bottom
        eta < tol
    else  # :top
        eta > (1 - tol)
    end
    
    return on_side ? Complex{FT}(1.0) : Complex{FT}(0.0)
end

"""
    compute_voltage(dist::CustomDistribution{FT,F}, edge_center, port_params) -> Complex{FT}

Compute voltage weight using the custom function.

The function receives normalized coordinates (ξ, η) in [0, 1].
"""
function compute_voltage(
    dist::CustomDistribution{FT,F},
    edge_center::MVec3D{FT},
    port_params::NamedTuple
) where {FT<:Real, F<:Function}
    u, v, _ = _rectangular_port_local_coordinates(
        edge_center,
        port_params.center,
        port_params.widthDir,
        port_params.heightDir,
        port_params.normal
    )
    xi = u / port_params.width + FT(0.5)
    eta = v / port_params.height + FT(0.5)
    return Complex{FT}(dist.weight_func(xi, eta))
end
