include("ExcitationDistributions.jl")
include("ModeImpedance.jl")
include("DeltaGapPort.jl")
include("Port_utils.jl")
include("DeltaGapArrayPort.jl")  # Must come before RectangularWaveguidePort (used as base)
include("RectangularWaveguidePort.jl")

include("CurrentProbe.jl")
include("PortArray.jl")


is_voltage_port(::AntennaType) = false
is_voltage_port(::PlaneWave) = false
is_voltage_port(::AbstractAntennaArray) = false

is_voltage_port(::PortType) = true
