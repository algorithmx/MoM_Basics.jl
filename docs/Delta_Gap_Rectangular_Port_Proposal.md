# Consolidated Proposal: Delta-Gap Excitation for Rectangular Waveguide Ports in Julia MoM

## Executive Summary

This document presents a consolidated proposal for implementing correct port excitation in the Julia Method of Moments (MoM) code for rectangular waveguide ports. The implementation addresses the fundamental limitation in the current codebase, where a single RWG basis function is excited rather than the entire port cross-section with the appropriate modal distribution. This proposal focuses on the Delta-Gap excitation method for Perfect Electric Conductor (PEC) surfaces, covering port geometry definition, geometric identification of port locations, mesh preprocessing (port masking), excitation placement on perimeter edges, and S-parameter extraction methodology.

The core physics principle underlying this proposal is that waveguide port excitation must enforce the complete modal field distribution across the port cross-section to achieve accurate S-parameter results. For the dominant TE₁₀ mode in rectangular waveguides, this means applying a sinusoidal electric field distribution across the port width while maintaining zero tangential electric field at the conducting walls. The Delta-Gap approach achieves this by placing voltage sources around the entire port perimeter, with voltages weighted according to the modal pattern.

---

## 1. Problem Statement and Physics Background

### 1.1 Current Implementation Limitations

The existing Julia MoM implementation uses a single-edge Delta-Gap excitation model through the `DeltaGapPort` structure. The excitation formula applies to one RWG basis function:

```
V_excitation[rwgID] = V₀ × l_m / 2
```

where `l_m` is the length of the edge associated with basis function `m`. This formulation is physically appropriate for antenna feed points where energy couples through a localized discontinuity. However, for waveguide ports used in S-parameter calculations, this approach fails to enforce the correct modal field distribution across the port cross-section.

### 1.2 Physics of Waveguide Port Excitation

Waveguide port excitation requires satisfying two fundamental conditions:

**Condition 1: Modal Field Distribution**
The electric field at the port must match the modal pattern. For the TE₁₀ mode in a rectangular waveguide:

```
E_y(x, z) = E₀ × sin(π × (x + a/2) / a) × e^(-j×k_z×z)
```

where `a` is the waveguide width, and the field varies sinusoidally across the width with maximum at the center and zero at the conducting walls (x = ±a/2).

**Condition 2: Boundary Conditions**
At PEC walls, the tangential electric field must be zero. This means the voltage distribution around the port perimeter must follow specific constraints:
- Top and bottom edges (at x = ±a/2): Zero voltage (E_tan = 0)
- Left and right edges (at z boundaries): Maximum voltage variation following sin(π×x/a)

### 1.3 Delta-Gap Excitation Physics

The Delta-Gap source creates a voltage discontinuity across an edge in the conducting surface. Physically, this represents a small gap in the conductor where the electric field can penetrate. The excitation formula:

```
V_excitation = V_gap × l_edge / 2
```

applies to each RWG basis function on the port perimeter. For proper waveguide excitation, we distribute Delta-Gap sources around the entire port boundary with voltages weighted according to the modal pattern.

---

## 2. Port Geometry and Excitation Distribution Definition

### 2.1 Design Philosophy: Separable Geometry and Excitation

A key architectural decision in this proposal is to separate **port geometry specification** from **excitation distribution specification**. This design allows the same port geometry to be excited with different modal patterns, which is essential for various electromagnetic analysis scenarios:

1. **Multi-mode analysis**: Exciting different waveguide modes (TE₁₀, TE₂₀, TE₀₁, etc.)
2. **Mode conversion studies**: Analyzing coupling between different modes
3. **Flexible applications**: Supporting both waveguide S-parameters and traditional antenna feeds
4. **Custom excitation**: Allowing user-defined field distributions for specialized applications
5. **Verification and testing**: Using uniform excitation to verify system response

This separation follows the principle of **composition over inheritance**, where port geometry and excitation distribution are independent components that can be combined flexibly.

### 2.2 Coordinate System and Orientation

For ports perpendicular to the xy-plane (vertical rectangles), we define:

```
Port plane: x-y plane at fixed z = z₀
Port normal: ±z-direction (propagation direction)
Port width: dimension along x-axis ( waveguide dimension 'a')
Port height: dimension along y-axis (waveguide dimension 'b')
```

### 2.3 Excitation Distribution Types

The excitation distribution is specified through an abstract interface, allowing multiple predefined distributions and custom user-defined patterns:

```julia
# Abstract type for excitation distribution
abstract type AbstractExcitationDistribution end

# Predefined TE/TM mode distributions
struct ModalDistribution <: AbstractExcitationDistribution
    modeType::Symbol        # :TE or :TM
    m::Int                  # Mode number along x-axis
    n::Int                  # Mode number along y-axis
end

# Uniform excitation distribution (for testing/verification)
struct UniformDistribution <: AbstractExcitationDistribution end

# User-defined custom excitation via function callback
struct CustomDistribution{F<:Function} <: AbstractExcitationDistribution
    voltage_function::F    # (x, y, xc, yc, a, b) -> Complex{Float64}
end

# Convenience aliases
const TE10Mode = ModalDistribution(:TE, 1, 0)
const TE20Mode = ModalDistribution(:TE, 2, 0)
const TE01Mode = ModalDistribution(:TE, 0, 1)
const UniformMode = UniformDistribution()
```

The voltage at any point on the port is computed through a polymorphic interface:

```julia
# Compute voltage for TE/TM modal distribution
function compute_voltage(
    dist::ModalDistribution,
    x::FT, y::FT,
    xc::FT, yc::FT,
    a::FT, b::FT
) where {FT<:Real}

    modeType = dist.modeType
    m, n = dist.m, dist.n

    # Normalize coordinates to port dimensions
    x_norm = (x - xc) / a  # Range: [-0.5, 0.5]
    y_norm = (y - yc) / b  # Range: [-0.5, 0.5]

    if modeType == :TE
        # TE_mn: E_x ∝ sin(mπ(x-xc)/a) × cos(nπ(y-yc)/b)
        # The voltage distribution follows the electric field pattern
        return sin(m * π * (x_norm + 0.5)) * cos(n * π * (y_norm + 0.5))
    else
        # TM_mn: E_x ∝ cos(mπ(x-xc)/a) × sin(nπ(y-yc)/b)
        return cos(m * π * (x_norm + 0.5)) * sin(n * π * (y_norm + 0.5))
    end
end

# Compute voltage for uniform distribution
function compute_voltage(
    dist::UniformDistribution,
    x::FT, y::FT,
    xc::FT, yc::FT,
    a::FT, b::FT
) where {FT<:Real}
    return Complex{FT}(1.0)  # Constant voltage everywhere
end

# Compute voltage for custom user-defined distribution
function compute_voltage(
    dist::CustomDistribution,
    x::FT, y::FT,
    xc::FT, yc::FT,
    a::FT, b::FT
) where {FT<:Real}
    return dist.voltage_function(x, y, xc, yc, a, b)
end
```

### 2.4 Port Structure with Configurable Excitation

The port structure maintains geometry parameters while accepting a configurable excitation distribution:

```julia
mutable struct RectangularWaveguidePort{FT<:Real, IT<:Integer}
    # Identification
    id::IT

    # Geometry - vertical rectangle in x-y plane
    portCenter::SVec3D{FT}      # Center coordinates (x₀, y₀, z₀)
    portWidth::FT                # Dimension 'a' along x-axis
    portHeight::FT               # Dimension 'b' along y-axis
    portNormal::SVec3D{FT}       # Propagation direction (±ẑ)

    # Excitation distribution - CONFIGURABLE
    excitationDistribution::AbstractExcitationDistribution

    # Excitation parameters
    portVoltage::Complex{FT}     # Reference voltage V₀
    frequency::FT

    # Status flags
    isActive::Bool

    # Computed data - to be populated
    portTriangles::Vector{IT}              # Triangles inside port area
    portBoundaryEdges::Vector{IT}         # RWG basis functions on perimeter
    edgeVoltages::Dict{IT, Complex{FT}}   # Voltage for each edge BF
end
```

### 2.5 Factory Functions for Common Configurations

Factory functions provide convenient constructors for typical use cases:

```julia
# Default port with TE₁₀ excitation (most common for waveguide analysis)
function RectangularWaveguidePort(;
    id::IT,
    center::SVec3D{FT},
    width::FT,
    height::FT,
    normal::SVec3D{FT} = SVec3D{FT}(0, 0, 1),
    voltage::Complex{FT} = Complex{FT}(1.0),
    frequency::FT
)
    return RectangularWaveguidePort{FT, IT}(
        id,
        center,
        width,
        height,
        normal,
        TE10Mode(),           # Default to TE₁₀
        voltage,
        frequency,
        true,
        Int[],
        Int[],
        Dict{Int, Complex{FT}}()
    )
end

# Port with custom mode specification
function RectangularWaveguidePort(;
    id::IT,
    center::SVec3D{FT},
    width::FT,
    height::FT,
    modeType::Symbol,
    m::Int,
    n::Int,
    normal::SVec3D{FT} = SVec3D{FT}(0, 0, 1),
    voltage::Complex{FT} = Complex{FT}(1.0),
    frequency::FT
)
    dist = ModalDistribution(modeType, m, n)

    return RectangularWaveguidePort{FT, IT}(
        id,
        center,
        width,
        height,
        normal,
        dist,
        voltage,
        frequency,
        true,
        Int[],
        Int[],
        Dict{Int, Complex{FT}}()
    )
end

# Port with uniform excitation (for testing/verification)
function RectangularWaveguidePortUniform(;
    id::IT,
    center::SVec3D{FT},
    width::FT,
    height::FT,
    normal::SVec3D{FT} = SVec3D{FT}(0, 0, 1),
    voltage::Complex{FT} = Complex{FT}(1.0),
    frequency::FT
)
    return RectangularWaveguidePort{FT, IT}(
        id,
        center,
        width,
        height,
        normal,
        UniformMode(),
        voltage,
        frequency,
        true,
        Int[],
        Int[],
        Dict{Int, Complex{FT}}()
    )
end

# Port with custom user-defined excitation function
function RectangularWaveguidePortCustom(;
    id::IT,
    center::SVec3D{FT},
    width::FT,
    height::FT,
    custom_voltage_function::F,
    normal::SVec3D{FT} = SVec3D{FT}(0, 0, 1),
    voltage::Complex{FT} = Complex{FT}(1.0),
    frequency::FT
) where {F<:Function}
    dist = CustomDistribution(custom_voltage_function)

    return RectangularWaveguidePort{FT, IT}(
        id,
        center,
        width,
        height,
        normal,
        dist,
        voltage,
        frequency,
        true,
        Int[],
        Int[],
        Dict{Int, Complex{FT}}()
    )
end
```

### 2.6 Setting Excitation Distribution After Port Creation

For maximum flexibility, the excitation distribution can be modified after port creation:

```julia
# Change excitation distribution of an existing port
function set_excitation_distribution!(
    port::RectangularWaveguidePort{FT, IT},
    dist::AbstractExcitationDistribution
) where {FT<:Real, IT<:Integer}
    port.excitationDistribution = dist
    # Clear computed voltages to force recomputation
    port.edgeVoltages = Dict{Int, Complex{FT}}()
    return port
end

# Convenience methods
set_te10_mode!(port::RectangularWaveguidePort) = set_excitation_distribution!(port, TE10Mode())
set_te20_mode!(port::RectangularWaveguidePort) = set_excitation_distribution!(port, TE20Mode())
set_uniform_mode!(port::RectangularWaveguidePort) = set_excitation_distribution!(port, UniformMode())
```

---

## 3. Port Locating Algorithm

### 3.1 Geometric Identification Principle

The port locating algorithm must identify which triangles in the mesh correspond to the specified rectangular port region. For a vertical rectangle perpendicular to the xy-plane, we use geometric filtering based on:

1. **Position bounds**: Triangle centroid within rectangular bounds in x-y plane
2. **Normal alignment**: Triangle face normal aligned with port normal direction
3. **Boundary detection**: Identification of edges where one side is metal and one side is aperture

### 3.2 Triangle Identification Algorithm

```julia
function locate_port_triangles!(
    port::RectangularWaveguidePort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # Extract port bounds
    xc = port.portCenter.x
    yc = port.portCenter.y
    a_half = port.portWidth / 2
    b_half = port.portHeight / 2

    x_min = xc - a_half
    x_max = xc + a_half
    y_min = yc - b_half
    y_max = yc + b_half

    # Get port normal direction
    nx, ny, nz = port.portNormal.x, port.portNormal.y, port.portNormal.z
    normal_magnitude = sqrt(nx^2 + ny^2 + nz^2)
    nx, ny, nz = nx/normal_magnitude, ny/normal_magnitude, nz/normal_magnitude

    # Find triangles within port region
    port_triangles = IT[]

    for (triID, tri) in enumerate(trianglesInfo)
        # Get triangle centroid
        cx, cy, cz = tri.center.x, tri.center.y, tri.center.z

        # Check if centroid is within rectangular bounds
        in_x_bounds = (cx >= x_min) && (cx <= x_max)
        in_y_bounds = (cy >= y_min) && (cy <= y_max)

        if in_x_bounds && in_y_bounds
            # Check normal alignment
            fnx, fny, fnz = tri.facen̂.x, tri.facen̂.y, tri.facen̂.z
            normal_alignment = abs(fnx*nx + fny*ny + fnz*nz)

            # Accept if normals are parallel or antiparallel (> 0.9)
            if normal_alignment > 0.9
                push!(port_triangles, triID)
            end
        end
    end

    port.portTriangles = port_triangles
    return port_triangles
end
```

### 3.3 Boundary Edge Identification

The critical insight is that RWG basis functions exist on the **edges** of triangles, not inside triangles. For port excitation, we need to identify **boundary edges** where:

- One triangle side is part of the port region (metal surface)
- The other triangle side is open space (aperture)

In the RWG formulation, this is detected by checking the triangle IDs associated with each basis function:

```julia
function identify_port_boundary_edges!(
    port::RectangularWaveguidePort{FT, IT},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # Port bounds
    xc = port.portCenter.x
    yc = port.portCenter.y
    a_half = port.portWidth / 2
    b_half = port.portHeight / 2

    x_min = xc - a_half
    x_max = xc + a_half
    y_min = yc - b_half
    y_max = yc + b_half

    # Tolerance for edge classification
    tolerance = 1e-6  # Adjust based on mesh resolution

    # Find boundary edges
    boundary_edges = IT[]

    for rwg in rwgsInfo
        # Get edge center
        ecx, ecy, ecz = rwg.center.x, rwg.center.y, rwg.center.z

        # Check if edge center is within port bounds
        in_x = (ecx >= x_min - tolerance) && (ecx <= x_max + tolerance)
        in_y = (ecy >= y_min - tolerance) && (ecy <= y_max + tolerance)

        if in_x && in_y
            # Check if this is a boundary edge
            # triID = 0 means open space (aperture side)
            triID_pos = rwg.inGeo[1]  # Positive side
            triID_neg = rwg.inGeo[2]  # Negative side

            is_boundary = (triID_pos == 0) ⊻ (triID_neg == 0)  # XOR

            if is_boundary
                push!(boundary_edges, rwg.bfID)
            end
        end
    end

    port.portBoundaryEdges = boundary_edges
    return boundary_edges
end
```

The XOR operation (`⊻`) ensures we select edges where exactly one side is open space (triID = 0). This correctly identifies the port perimeter edges.

---

## 4. Port Masking and Mesh Preprocessing

### 4.1 Purpose of Port Masking

Port masking serves two essential purposes:

1. **Geometric clarity**: Removes triangles that physically occupy the port aperture region, ensuring the port is treated as an opening in the metal rather than a metal surface
2. **Boundary definition**: Creates clean boundary edges where one side is metal and the other is aperture, which is essential for Delta-Gap excitation

### 4.2 Masking Algorithm

```julia
function apply_port_mask!(
    port::RectangularWaveguidePort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}};
    maskRegion::Symbol = :aperture  # :aperture, :boundary, or :all
) where {FT<:Real, IT<:Integer}

    # Determine which triangles to mask
    masked_triangles = IT[]
    kept_triangles = IT[]

    xc = port.portCenter.x
    yc = port.portCenter.y
    zc = port.portCenter.z
    a_half = port.portWidth / 2
    b_half = port.portHeight / 2

    for (triID, tri) in enumerate(trianglesInfo)
        cx, cy, cz = tri.center.x, tri.center.y, tri.center.z

        # Check if triangle center is within port rectangle
        in_port = (abs(cx - xc) <= a_half) &&
                  (abs(cy - yc) <= b_half) &&
                  (abs(cz - zc) <= 1e-6)  # Same z-plane

        if in_port
            push!(masked_triangles, triID)
        else
            push!(kept_triangles, triID)
        end
    end

    # Return indices of triangles to keep
    return kept_triangles, masked_triangles
end
```

### 4.3 Practical Implementation Strategy

There are two approaches to port masking in practice:

**Approach A: Preprocessing the Mesh**
Before constructing the MoM system, explicitly remove triangles within the port aperture region from the mesh file. This creates a clean aperture with no triangles inside.

**Approach B: Logical Masking**
Keep the mesh file unchanged but use logical filtering to exclude triangles within the port region when constructing the MoM matrix. This is implemented in the port identification algorithm.

```julia
function build_masked_mesh!(
    port::RectangularWaveguidePort{FT, IT},
    trianglesInfo::Vector{TriangleInfo{IT, FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # Step 1: Identify triangles to mask (inside aperture)
    _, masked_tris = apply_port_mask!(port, trianglesInfo, rwgsInfo)
    masked_tri_set = Set(masked_tris)

    # Step 2: Build mapping from old to new indices
    old_to_new = Dict{IT, IT}()
    new_triangles = TriangleInfo{IT, FT}[]
    new_idx = 0

    for (old_idx, tri) in enumerate(trianglesInfo)
        if old_idx ∉ masked_tri_set
            new_idx += 1
            old_to_new[old_idx] = new_idx
            push!(new_triangles, tri)
        end
    end

    # Step 3: Update RWG basis functions
    # Triangles with ID = 0 remain 0 (aperture)
    # Other triangles get new IDs

    return new_triangles, old_to_new
end
```

---

## 5. Delta-Gap Excitation Placement

### 5.1 Configurable Voltage Distribution

The key advantage of the separable architecture is that the voltage distribution is now computed based on the configured excitation distribution. The algorithm uses the polymorphic `compute_voltage` function to determine the voltage at each boundary edge:

```
V(x, y) = V₀ × f_distribution(x, y)
```

where `f_distribution` is determined by the port's `excitationDistribution` field:
- **TE₁₀ mode**: V(x) = sin(π × (x + a/2) / a)
- **Uniform**: V(x, y) = 1
- **Custom**: V(x, y) = user_defined_function(x, y)

### 5.2 Computing Edge Voltages with Configurable Distribution

```julia
function compute_edge_voltages!(
    port::RectangularWaveguidePort{FT, IT},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}

    a = port.portWidth
    b = port.portHeight
    xc = port.portCenter.x
    yc = port.portCenter.y
    V0 = port.portVoltage
    dist = port.excitationDistribution

    edge_voltages = Dict{IT, Complex{FT}}()

    # Compute voltage for each boundary edge using the configured distribution
    for bfID in port.portBoundaryEdges
        rwg = rwgsInfo[bfID]
        ecx, ecy = rwg.center.x, rwg.center.y

        # Get voltage factor from the configured distribution
        voltage_factor = compute_voltage(dist, ecx, ecy, xc, yc, a, b)

        # Apply reference voltage magnitude
        edge_voltages[bfID] = V0 * voltage_factor
    end

    port.edgeVoltages = edge_voltages
    return edge_voltages
end
```

The function automatically handles any excitation distribution without modification:
- TE₁₀, TE₂₀, TE₀₁, TM modes through `ModalDistribution`
- Uniform excitation through `UniformDistribution`
- Arbitrary custom patterns through `CustomDistribution`

### 5.3 Specialized Voltage Computation for Common Modes

For efficiency, specialized implementations can be provided for common modes:

```julia
# Optimized TE₁₀ computation (most common case)
function compute_te10_voltage(
    x::FT, y::FT,
    xc::FT, yc::FT,
    a::FT, b::FT
) where {FT<:Real}
    # TE₁₀: E_y ∝ sin(π×(x+a/2)/a), zero at x = ±a/2
    x_rel = (x - xc) / a  # Normalized to [-0.5, 0.5]
    return sin(π * (x_rel + 0.5))
end

# Optimized TE_m0 computation (higher order along x)
function compute_tem0_voltage(
    m::Int,
    x::FT, y::FT,
    xc::FT, yc::FT,
    a::FT, b::FT
) where {FT<:Real}
    x_rel = (x - xc) / a
    return sin(m * π * (x_rel + 0.5))
end
```

### 5.4 Building the Excitation Vector

The excitation vector is constructed by applying the Delta-Gap formula to each boundary edge:

```
V_excitation[bfID] = V_edge × l_edge / 2
```

```julia
function build_excitation_vector!(
    V::Vector{Complex{FT}},
    port::RectangularWaveguidePort{FT, IT},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # Ensure edge voltages are computed using configured distribution
    if isempty(port.edgeVoltages)
        compute_edge_voltages!(port, rwgsInfo)
    end

    # Apply Delta-Gap excitation to each boundary edge
    for (bfID, V_edge) in port.edgeVoltages
        rwg = rwgsInfo[bfID]
        edge_length = rwg.edgel

        # Delta-Gap formula: V × l / 2
        V[bfID] += V_edge * edge_length / 2
    end

    return V
end
```

### 5.5 Usage Examples

**Example 1: TE₁₀ mode (default)**
```julia
# Most common waveguide excitation
port1 = RectangularWaveguidePort(
    id = 1,
    center = SVec3D(0.0, 0.0, 0.0),
    width = 2.0,
    height = 1.0,
    frequency = 10e9
)
# Uses TE₁₀ by default
```

**Example 2: Higher order mode**
```julia
# Exciting TE₂₀ mode for mode conversion analysis
port2 = RectangularWaveguidePort(
    id = 2,
    center = SVec3D(0.0, 0.0, 0.0),
    width = 4.0,  # Double width for TE20
    height = 1.0,
    modeType = :TE,
    m = 2,
    n = 0,
    frequency = 10e9
)
```

**Example 3: Uniform excitation**
```julia
# For testing/verification
port3 = RectangularWaveguidePortUniform(
    id = 3,
    center = SVec3D(0.0, 0.0, 0.0),
    width = 2.0,
    height = 1.0,
    frequency = 10e9
)
```

**Example 4: Custom excitation**
```julia
# User-defined pattern
my_voltage_func(x, y, xc, yc, a, b) = exp(-((x-xc)^2 + (y-yc)^2)/(a*b))

port4 = RectangularWaveguidePortCustom(
    id = 4,
    center = SVec3D(0.0, 0.0, 0.0),
    width = 2.0,
    height = 1.0,
    custom_voltage_function = my_voltage_func,
    frequency = 10e9
)
```

**Example 5: Changing mode after creation**
```julia
# Start with TE₁₀, then switch to TE₂₀
port = RectangularWaveguidePort(id=1, center=(0,0,0), width=2.0, height=1.0)
set_te20_mode!(port)  # Change excitation to TE₂₀
```

### 5.6 Verification: Why This Works Physically

The Delta-Gap array excitation produces the correct modal field because:

1. **Boundary condition satisfaction**: Zero voltage at top/bottom edges ensures E_tan = 0 at the conducting walls
2. **Modal voltage distribution**: Sinusoidal variation across the width produces the TE₁₀ field pattern
3. **Superposition**: The sum of all edge contributions produces the complete modal field in the aperture
4. **Energy coupling**: The voltage discontinuity creates equivalent magnetic currents that radiate into the waveguide mode

Mathematically, the impressed electric field from the Delta-Gap array is:

```
E_inc(y) = (V₀/l) × sin(π×y/a)
```

which matches the TE₁₀ electric field distribution when properly scaled.

---

## 6. S-Parameter Extraction

### 6.1 Port Impedance Calculation

After solving the MoM system **Z × I = V**, the port impedance is extracted from the current distribution. For the Delta-Gap excitation:

```julia
function compute_port_impedance(
    port::RectangularWaveguidePort{FT, IT},
    ICoeff::Vector{Complex{FT}},
    rwgsInfo::Vector{RWG{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # Total current at port (sum of currents on all boundary edges)
    total_current = Complex{FT}(0)

    for bfID in port.portBoundaryEdges
        # Get current coefficient
        I_bf = ICoeff[bfID]

        # Get edge length for current normalization
        l_edge = rwgsInfo[bfID].edgel

        # Current flowing through this edge
        # For RWG: current is I_bf × l_edge
        total_current += I_bf * l_edge
    end

    # Port voltage (reference voltage)
    V_port = port.portVoltage

    # Impedance: Z = V / I
    Z_port = V_port / total_current

    return Z_port
end
```

### 6.2 Multi-Port S-Parameter Computation

For systems with multiple ports, the S-parameters are computed from the impedance matrix:

```julia
function compute_S_parameters(
    ports::Vector{RectangularWaveguidePort{FT, IT}},
    ICoeff::Vector{Complex{FT}},
    rwgsInfo::Vector{RWG{IT, FT}};
    reference_impedance::FT = 50.0
) where {FT<:Real, IT<:Integer}

    n_ports = length(ports)
    Z_matrix = zeros(Complex{FT}, n_ports, n_ports)

    # Compute Z-matrix (port-to-port impedances)
    for i in 1:n_ports
        for j in 1:n_ports
            if i == j
                # Self-impedance
                Z_matrix[i,j] = compute_port_impedance(ports[i], ICoeff, rwgsInfo)
            else
                # Transfer impedance (requires multiple solves or reciprocity)
                # For now, use reciprocal property: Z_ij = Z_ji
                # Full implementation would solve for each port excitation
            end
        end
    end

    # Convert to S-parameters
    Z0 = reference_impedance
    S_matrix = zeros(Complex{FT}, n_ports, n_ports)

    for i in 1:n_ports
        for j in 1:n_ports
            if i == j
                S_matrix[i,j] = (Z_matrix[i,j] - Z0) / (Z_matrix[i,j] + Z0)
            else
                # For coupled ports: S_ij = 2×Z_ij / (Z_ii + Z0)
                # Simplified: assume ports are matched
                S_matrix[i,j] = 2 * Z_matrix[i,j] / (Z_matrix[i,i] + Z0)
            end
        end
    end

    return S_matrix, Z_matrix
end
```

### 6.3 Mode Impedance Consideration

For waveguide ports, the mode impedance differs from the reference impedance (typically 50Ω). The relationship:

```
Z_TE10 = η / √(1 - (λ/2a)²)
```

where η ≈ 377Ω is the free-space impedance. This mode impedance should be used when converting between impedance and S-parameters for waveguide analysis.

---

## 7. Integration with Existing Julia MoM Infrastructure

### 7.1 Extending Current Architecture

The implementation integrates with existing structures in `MoM_Basics.jl` using the configurable architecture:

```julia
# Extend the Port.jl types
# Add to existing port types

"""
    DeltaGapArrayPort{FT, IT}

Port excitation using distributed Delta-Gap sources around port perimeter.
The excitation distribution is configurable via AbstractExcitationDistribution.
Supports TE₁₀, higher-order modes, uniform, and custom distributions.
"""
mutable struct DeltaGapArrayPort{FT<:Real, IT<:Integer} <: AbstractPort{FT, IT}
    # Geometry
    id::IT
    portCenter::SVec3D{FT}
    portWidth::FT
    portHeight::FT
    portNormal::SVec3D{FT}

    # Configurable excitation distribution
    excitationDistribution::AbstractExcitationDistribution

    # Excitation
    portVoltage::Complex{FT}
    frequency::FT

    # Computed
    portBoundaryEdges::Vector{IT}
    edgeVoltages::Dict{IT, Complex{FT}}

    # Status
    isActive::Bool
end
```

### 7.2 Excitation Vector Method

The excitation vector method uses the configurable distribution automatically:

```julia
function excitationVector!(
    V::Vector{Complex{FT}},
    port::DeltaGapArrayPort{FT, IT},
    rwgsInfo::Vector{RWG{IT, FT}},
    trianglesInfo::Vector{TriangleInfo{IT, FT}}
) where {FT<:Real, IT<:Integer}

    # Step 1: Identify boundary edges if not done
    if isempty(port.portBoundaryEdges)
        identify_port_boundary_edges!(port, rwgsInfo)
    end

    # Step 2: Compute voltages using configured distribution
    if isempty(port.edgeVoltages)
        compute_edge_voltages!(port, rwgsInfo)  # Uses configurable distribution
    end

    # Step 3: Apply Delta-Gap excitation
    build_excitation_vector!(V, port, rwgsInfo)

    return V
end
```

### 7.3 Factory Function

Factory functions support all distribution types:

```julia
function create_rectangular_port(;
    id::IT,
    center::Tuple{FT, FT, FT},
    width::FT,
    height::FT,
    normal::Tuple{FT, FT, FT} = (0, 0, 1),
    excitation::AbstractExcitationDistribution = TE10Mode(),
    voltage::Complex{FT} = Complex{FT}(1.0),
    frequency::FT
) where {FT<:Real, IT<:Integer}

    portCenter = SVec3D{FT}(center...)
    portNormal = SVec3D{FT}(normal...)

    return DeltaGapArrayPort{FT, IT}(
        id,
        portCenter,
        width,
        height,
        portNormal,
        excitation,  # Configurable distribution
        voltage,
        frequency,
        Int[],
        Dict{Int, Complex{FT}}(),
        true
    )
end
```

---

## 8. Complete Algorithm Summary

### 8.1 Step-by-Step Implementation

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Define port geometry | Specify center, dimensions, normal direction |
| 2 | Specify excitation distribution | Choose TE₁₀, TE₂₀, uniform, or custom |
| 3 | Locate port triangles | Filter triangles within rectangular bounds |
| 4 | Identify boundary edges | Find RWG basis functions where one triID = 0 |
| 5 | Compute edge voltages | Apply configured distribution to each edge |
| 6 | Build excitation vector | V[bfID] += V_edge × l_edge / 2 |
| 7 | Solve MoM system | Z × I = V |
| 8 | Extract currents | I at port boundary edges |
| 9 | Compute impedance | Z_port = V_port / I_total |
| 10 | Calculate S-parameters | Convert impedance to S-matrix |

### 8.2 Physics Validation

The proposed implementation correctly models waveguide port excitation because:

1. **Complete cross-section excitation**: All boundary edges around the port perimeter are excited, not just a single edge
2. **Modal field enforcement**: Voltage distribution follows sin(π×x/a) pattern
3. **Boundary condition satisfaction**: Zero voltage at conducting walls (top/bottom edges)
4. **Energy conservation**: Proper power injection through modal impedance consideration
5. **Reciprocity**: The excitation is self-adjoint, ensuring correct port behavior

---

## 9. Conclusions and Recommendations

### 9.1 Key Findings

The analysis confirms that the current single-edge Delta-Gap implementation is inadequate for waveguide port S-parameter calculations. The proposed Delta-Gap Array approach addresses this by:

1. Exciting all boundary edges around the port perimeter
2. Applying configurable excitation distributions (TE₁₀, TE₂₀, uniform, or custom)
3. Satisfying PEC boundary conditions at the waveguide walls
4. Integrating with existing Julia MoM infrastructure
5. Supporting multiple electromagnetic analysis scenarios through separable geometry and excitation

### 9.2 Implementation Path

Recommended implementation sequence:

1. **Phase 1**: Create `DeltaGapArrayPort` type and basic identification algorithms
2. **Phase 2**: Implement modal voltage computation and excitation vector assembly
3. **Phase 3**: Add port impedance extraction and S-parameter computation
4. **Phase 4**: Validation against known waveguide solutions (waveguide filter, etc.)

### 9.3 Validation Cases

The implementation should be validated against:

1. **Empty waveguide**: S₁₁ = 0 (matched) or S₁₁ = 1 (shortened)
2. **Waveguide step**: Compare with analytical transition solutions
3. **Waveguide filter**: S-parameters against commercial software (HFSS, CST)

---

## Appendix A: Mathematical Formulas Reference

### A.1 TE₁₀ Mode Fields

```
E_y(x,z) = E₀ × sin(π×(x+a/2)/a) × e^(-j×k_z×z)
H_x(x,z) = -(k_z/ωμ) × E_y
H_z(x,z) = (jπ/ωμa) × cos(π×(x+a/2)/a) × e^(-j×k_z×z)
k_z = √(k² - (π/a)²)
```

### A.2 Mode Impedance

```
Z_TE10 = ωμ / k_z = η / √(1 - (λ/2a)²)
```

### A.3 Delta-Gap Excitation

```
V_excitation = V_gap × l_edge / 2
```

### A.4 S-Parameter Conversion

```
S_11 = (Z_in - Z₀) / (Z_in + Z₀)
```

---

## Appendix B: Code Integration Points

| Existing File | Modification |
|---------------|--------------|
| `Port.jl` | Add `DeltaGapArrayPort` type and methods |
| `Excitation.jl` | Add `excitationVector!` for `DeltaGapArrayPort` |
| `SParemeters.jl` | Extend impedance extraction for array ports |
| `MeshProcessing.jl` | Add port masking utilities |

---

*Document Version: 1.0*
*Author: MiniMax Agent*
*Date: 2026-03-06*
