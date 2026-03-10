include("DeltaGapPort.jl")
include("Port_utils.jl")
include("RectangularWaveguidePort.jl")

is_voltage_port(::ExcitingSource) = false
is_voltage_port(::DeltaGapPort) = true
is_voltage_port(::RectangularWaveguidePort) = true

include("CurrentProbe.jl")
include("PortArray.jl")

