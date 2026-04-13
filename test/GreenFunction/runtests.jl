# GreenFunction Module Test Suite
# Tests for mathematical correctness of Green's function implementations
#
# References:
# - layered_media_gf_research_materials.md (Section 1.1, 1.2)
# - mom_ground_plane_dielectric_approaches.md (Section 1.1, 1.2)
# - ground_plane_implementation_proposal.md (Section 3.2)

using Test
using MoM_Basics
using MoM_Basics.GreenFunction
using StaticArrays: SVector
using LinearAlgebra: norm

# Run all test files
@testset "GreenFunction Module Tests" begin
    include("test_FreeSpaceGF.jl")
    include("test_GroundPlaneGF.jl")
    include("test_LayerStack.jl")
    include("test_configuration.jl")
    include("test_evaluate_greenfunc_star.jl")
    # DCIM z-dependence tests (separate due to longer runtime)
    # include("test_DCIM_z_dependence.jl")  # Commented out - run manually
end
