include("ExcitationDistributions.jl")
include("ModeImpedance.jl")
include("DeltaGapPort.jl")
include("Port_utils.jl")
include("DeltaGapArrayPort.jl")  # Must come before RectangularEdgePort (used as base)
include("RectangularEdgePort.jl")
include("PortMasking.jl")  # Logical port masking without mesh modification

include("CurrentProbe.jl")
include("PortArray.jl")


is_voltage_port(::AntennaType) = false
is_voltage_port(::PlaneWave) = false
is_voltage_port(::AbstractAntennaArray) = false

is_voltage_port(::PortType) = true
