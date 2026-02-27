# MoM_Basics.jl - Infrastructure Layer

Problem definition and data structures. Provider module for MoM_Kernels.jl and MoM_Visualizing.jl.

## Source Tree

```
src/
├── MoM_Basics.jl           # Main module, exports public API
├── BasicStuff.jl           # Core types (Vec3D, SVec3D, θϕInfo)
├── CoorTrans.jl            # Coordinate transformations
├── GaussQuadrature4Geos.jl # Gaussian quadrature rules for geometries
├── ParametersSet.jl        # Simulation parameters
├── Inputs.jl               # I/O handling
├── MeshAndBFs.jl           # Mesh-basis function coordination
├── UsefulFunctions.jl      # Utilities
├── Recorder.jl             # Memory & timing profiler
│
├── Sources/                # Excitation sources
│   ├── Source.jl           # Abstract base (ExcitingSource, AbstractIntegralEquation)
│   ├── Port.jl             # DeltaGapPort, CurrentProbe, S-parameter calc
│   ├── Planewave.jl        # Plane wave excitation
│   ├── MagneticDipole.jl   # Magnetic dipole source
│   ├── AntettaArray.jl     # Antenna array definitions
│   └── FieldExtraction.jl  # Field data extraction
│
├── BasisFunctions/         # Basis function implementations
│   ├── RWG.jl              # Rao-Wilton-Glisson surface basis
│   ├── SWG.jl              # Schaubert-Wilton-Glisson volume basis
│   ├── PWC.jl              # Piecewise constant basis
│   ├── RBF.jl              # Hexahedral RWG basis
│   └── BFs.jl              # Basis function interface
│
├── MeshProcess/            # Mesh processing
└── BasicVSCellType/        # Mesh cell type definitions
    ├── Triangles.jl
    ├── Tetrahedras.jl
    ├── Hexahedras.jl
    └── Quadrangle.jl
```

## Key Types

**Integral Equations**: `EFIE`, `MFIE`, `CFIE`

**Basis Functions**: `RWG`, `SWG`, `PWC`, `RBF`

**Excitation Sources**: `DeltaGapPort`, `CurrentProbe`, `PlaneWave`, `MagneticDipole`

**Mesh Cells**: `TriangleInfo`, `TetrahedraInfo`, `HexahedraInfo`
