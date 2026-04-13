# GreenFunction Module
# Unified interface for Green's functions in MoM simulations
# Supports: free-space, ground plane (image theory), and layered media (DCIM)

module GreenFunction

using StaticArrays, LinearAlgebra
using ..MoM_Basics: Vec3D, SVec3D, MVec3D, Precision, Params, GFParams

export AbstractGreenFunction, AbstractGFCache,
       FreeSpaceGF, GroundPlaneGF, LayeredMediaGF,
       GreenFuncVals,
       evaluate_greenfunc, evaluate_greenfunc_A, evaluate_greenfunc_phi,
       evaluate_greenfunc_direct, evaluate_greenfunc_image, evaluate_greenfunc_star,
       get_green_function_type, set_green_function_type!, set_layer_stack!, reset_green_function_config!, create_green_function,
       horizontal_distance, mirror_point_across_ground,
       evaluate_at_same_height, height_above_ground, is_above_ground,
       # Layer stack
       LayerStack, LayerInfo, DCIMCoefficients, ComplexImage,
       get_layer_index, get_layer, are_in_same_layer,
       layer_thickness_above, layer_thickness_below,
       is_halfspace, effective_permittivity,
       has_ground_plane, ground_plane_z,
       # GPOF
       GPOFResult, gpof_fit, gpof_fit_real, evaluate_gpof, 
       gpof_residual, gpof_relative_error, diagnose_gpof, is_stable, has_conjugate_pairs,
       # Spectral GF
       SpectralGF, compute_spectral_gf, sample_real_axis, sample_sommerfeld_branch_cut,
       # DCIM
       DCIMFittingResult, DCIMParameters, dcim_fit, evaluate_dcim, evaluate_dcim_smooth

# =============================================================================
# Abstract Types
# =============================================================================

"""
    AbstractGreenFunction{FT<:AbstractFloat}

Abstract base type for all Green's function implementations.

Subtypes:
- `FreeSpaceGF`: Free-space scalar Green's function (existing behavior)
- `GroundPlaneGF`: Image theory for PEC ground plane
- `LayeredMediaGF`: Layered media via DCIM
"""
abstract type AbstractGreenFunction{FT<:AbstractFloat} end

"""
    AbstractGFCache{FT<:AbstractFloat}

Abstract base type for Green's function precomputation caches.
Used for storing frequency-dependent coefficients (e.g., DCIM images).
"""
abstract type AbstractGFCache{FT<:AbstractFloat} end

# =============================================================================
# Green's Function Evaluation Result Type
# =============================================================================

"""
    GreenFuncVals{CT<:Complex}

Container for vector and scalar potential Green's function values.

Fields:
- `g_A::CT`: Vector potential Green's function (dyadic xx/yy component)
- `g_phi::CT`: Scalar potential Green's function

Note: In free space, g_A = g_phi. In layered media, they differ.
"""
struct GreenFuncVals{CT<:Complex}
    g_A::CT
    g_phi::CT
end

# Convenience constructor for free-space (same value)
GreenFuncVals{CT}(g::CT) where {CT<:Complex} = GreenFuncVals{CT}(g, g)

# =============================================================================
# Core Interface Functions (to be implemented by subtypes)
# =============================================================================

"""
    evaluate_greenfunc(gf::AbstractGreenFunction, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate the Green's function between observation point `r_obs` and source point `r_src`.

Returns a `GreenFuncVals` containing both vector potential (g_A) and scalar potential (g_phi)
components.

# Arguments
- `gf::AbstractGreenFunction`: The Green's function instance
- `r_obs::AbstractVector`: Observation point coordinates (3D)
- `r_src::AbstractVector`: Source point coordinates (3D)

# Returns
- `GreenFuncVals`: Container with g_A and g_phi values

# Notes
- This is the primary interface for Z-matrix assembly.
- Dispatches to type-specific implementations.
- In free space: g_A = g_phi = exp(-jkR)/R
- With ground plane: g_A and g_phi have different image signs
- In layered media: g_A and g_phi are different functions
"""
function evaluate_greenfunc end

"""
    evaluate_greenfunc_A(gf::AbstractGreenFunction, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate only the vector potential Green's function component.

# Returns
- `Complex`: g_A value

# Notes
- Convenience wrapper that extracts g_A from `evaluate_greenfunc`
- Use when only vector potential is needed
"""
function evaluate_greenfunc_A(gf::AbstractGreenFunction{FT}, 
                               r_obs::AbstractVector, 
                               r_src::AbstractVector) where {FT}
    vals = evaluate_greenfunc(gf, r_obs, r_src)
    return vals.g_A
end

"""
    evaluate_greenfunc_phi(gf::AbstractGreenFunction, r_obs::AbstractVector, r_src::AbstractVector)

Evaluate only the scalar potential Green's function component.

# Returns
- `Complex`: g_phi value

# Notes
- Convenience wrapper that extracts g_phi from `evaluate_greenfunc`
- Use when only scalar potential is needed
"""
function evaluate_greenfunc_phi(gf::AbstractGreenFunction{FT}, 
                                 r_obs::AbstractVector, 
                                 r_src::AbstractVector) where {FT}
    vals = evaluate_greenfunc(gf, r_obs, r_src)
    return vals.g_phi
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    horizontal_distance(r1::AbstractVector, r2::AbstractVector)

Compute horizontal (xy-plane) distance between two points.

# Arguments
- `r1, r2`: 3D coordinates

# Returns
- `FT`: sqrt((x1-x2)² + (y1-y2)²)

# Notes
- Used in layered media where horizontal and vertical separations are treated differently
"""
function horizontal_distance(r1::AbstractVector{FT}, r2::AbstractVector{FT}) where {FT<:AbstractFloat}
    return sqrt((r1[1]-r2[1])^2 + (r1[2]-r2[2])^2)
end

function horizontal_distance(r1::Vec3D{FT}, r2::Vec3D{FT}) where {FT<:AbstractFloat}
    return sqrt((r1[1]-r2[1])^2 + (r1[2]-r2[2])^2)
end

"""
    mirror_point_across_ground(r::AbstractVector, z_gnd::Real)

Mirror a point across a ground plane at z = z_gnd.

# Arguments
- `r`: Original point coordinates (3D)
- `z_gnd`: Ground plane z-coordinate

# Returns
- `SVec3D`: Mirrored point coordinates

# Notes
- Used in image theory for PEC ground plane
- x, y coordinates unchanged; z reflected: z_img = 2*z_gnd - z
"""
function mirror_point_across_ground(r::AbstractVector{FT}, z_gnd::FT) where {FT<:AbstractFloat}
    return SVec3D{FT}(r[1], r[2], 2*z_gnd - r[3])
end

function mirror_point_across_ground(r::Vec3D{FT}, z_gnd::FT) where {FT<:AbstractFloat}
    return SVec3D{FT}(r[1], r[2], 2*z_gnd - r[3])
end

# =============================================================================
# Configuration Functions
# =============================================================================
# Note: Configuration is managed in ParametersSet.jl via GFParams global
# These functions are thin wrappers that delegate to the central configuration

"""
    set_green_function_type!(gf_type::Symbol; kwargs...)

Configure the Green's function type for simulation.

# Arguments
- `gf_type::Symbol`: 
  - `:freespace` - Free-space Green's function (default, backward compatible)
  - `:groundplane` - PEC ground plane via image theory
  - `:layered` - Layered media via DCIM
- `z_gnd::Real`: Ground plane z-coordinate (required for :groundplane)

# Examples
```julia
# Free space (default)
set_green_function_type!(:freespace)

# Ground plane at z = 0
set_green_function_type!(:groundplane; z_gnd=0.0)

# Layered media (requires set_layer_stack! first)
layers = [LayerInfo("substrate", 11.7, 1.0, 500e-6), LayerInfo("air", 1.0, 1.0, Inf)]
stack = LayerStack(layers; reference_z=0.0, has_ground_plane=true)
set_layer_stack!(stack)
set_green_function_type!(:layered)

# Reset to defaults
set_green_function_type!(:freespace)
```

# Notes
- This function updates the global GFParams configuration in ParametersSet.jl
- For `:layered`, you must call `set_layer_stack!(stack)` first to provide the layer stack
- The layer stack is stored in a global reference and used by `create_green_function()`
"""
function set_green_function_type!(gf_type::Symbol; kwargs...)
    # Update the GFParams global from ParametersSet.jl
    GFParams.gf_type = gf_type
    
    if gf_type == :freespace
        GFParams.has_layer_stack = false
        GFParams.ground_plane_z = Inf
        
    elseif gf_type == :groundplane
        z_gnd = get(kwargs, :z_gnd, Inf)
        if isinf(z_gnd)
            error("Ground plane requires finite z_gnd coordinate. Use: set_green_function_type!(:groundplane; z_gnd=0.0)")
        end
        GFParams.ground_plane_z = Float64(z_gnd)
        GFParams.has_layer_stack = false
        
    elseif gf_type == :layered
        if _prebuilt_layer_stack[] === nothing
            error("No layer stack configured. Call set_layer_stack!(stack) before set_green_function_type!(:layered).")
        end
        GFParams.has_layer_stack = true
        
    else
        error("Unknown Green's function type: $gf_type. Valid options: :freespace, :groundplane, :layered")
    end
    
    return nothing
end

"""
    get_green_function_type() -> Symbol

Get the currently configured Green's function type.
"""
get_green_function_type() = GFParams.gf_type

"""
    reset_green_function_config!()

Reset Green's function configuration to defaults (free-space).
"""
function reset_green_function_config!()
    GFParams.gf_type = :freespace
    GFParams.ground_plane_z = Inf
    GFParams.layer_stack_file = ""
    GFParams.has_layer_stack = false
    _prebuilt_layer_stack[] = nothing
    return nothing
end

# Prebuilt layer stack storage (avoids JSON3 dependency in MoM_Basics)
const _prebuilt_layer_stack = Ref{Any}(nothing)

"""
    set_layer_stack!(stack::LayerStack)

Store a pre-built LayerStack for use with `:layered` Green's function.

Call this before `set_green_function_type!(:layered)` to provide the
electromagnetic layer stack (the caller parses JSON and builds the LayerStack).

# Example
```julia
layers = [LayerInfo("substrate", 11.7, 1.0, 422e-6), LayerInfo("air", 1.0, 1.0, Inf)]
stack = LayerStack(layers; reference_z=0.0, has_ground_plane=true)
set_layer_stack!(stack)
set_green_function_type!(:layered)
```
"""
function set_layer_stack!(stack)
    _prebuilt_layer_stack[] = stack
    GFParams.has_layer_stack = true
    return nothing
end

"""
    create_green_function([gf_type::Symbol]) -> AbstractGreenFunction

Factory function to create the appropriate Green's function instance.

# Arguments
- `gf_type::Symbol`: Optional, defaults to configured type. One of:
  - `:freespace` - Returns `FreeSpaceGF`
  - `:groundplane` - Returns `GroundPlaneGF` (requires `z_gnd` set via `set_green_function_type!`)
  - `:layered` - Returns `LayeredMediaGF` (requires `set_layer_stack!` called first)

# Returns
- `AbstractGreenFunction`: Instance of appropriate concrete type

# Examples
```julia
# Free space
set_green_function_type!(:freespace)
gf = create_green_function()

# Ground plane
set_green_function_type!(:groundplane; z_gnd=0.0)
gf = create_green_function()

# Layered media
layers = [LayerInfo("substrate", 11.7, 1.0, 500e-6), LayerInfo("air", 1.0, 1.0, Inf)]
stack = LayerStack(layers; reference_z=0.0, has_ground_plane=true)
set_layer_stack!(stack)
set_green_function_type!(:layered)
gf = create_green_function()  # Returns LayeredMediaGF
```

# Notes
- Uses global GFParams for parameters
- For `:layered`, the layer stack must be pre-configured via `set_layer_stack!()`
- Should be called once per frequency (after Params is set)
- `LayeredMediaGF` performs DCIM fitting during construction, which may be expensive
"""
function create_green_function(gf_type::Symbol = GFParams.gf_type)
    FT = Precision.FT
    k = Params.K_0  # Free-space wavenumber
    
    if gf_type == :freespace
        return FreeSpaceGF{FT}(k)
        
    elseif gf_type == :groundplane
        return GroundPlaneGF{FT}(Complex{FT}(k), FT(GFParams.ground_plane_z))
        
    elseif gf_type == :layered
        stack = _prebuilt_layer_stack[]
        if stack === nothing
            error("No layer stack configured. Call set_layer_stack!(stack) first.")
        end
        freq = Params.frequency
        return LayeredMediaGF(stack, FT(freq))
        
    else
        error("Unknown Green's function type: $gf_type")
    end
end

# =============================================================================
# Include Concrete Implementations
# =============================================================================

include("LayerStack.jl")      # Layer stack data structures
include("GPOF.jl")            # Generalized Pencil of Function
include("SpectralGF.jl")      # Spectral domain Green's function
include("DCIM.jl")            # Discrete Complex Image Method
include("FreeSpaceGF.jl")     # Free-space Green's function
include("GroundPlaneGF.jl")   # Ground plane via image theory
include("LayeredMediaGF.jl")  # Layered media

end # module GreenFunction
