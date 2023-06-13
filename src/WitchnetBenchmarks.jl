module WitchnetBenchmarks

include("Utils.jl")
include("libmagds.jl")
include("predeval.jl")
include("plots.jl")
include("models.jl")
include("PMLB/PMLB.jl")
include("benchmarks/Benchmarks.jl")

end # module WitchnetBenchmark
# acceleration=CUDALibs()
# MLJ.default_resource(CPUProcesses())