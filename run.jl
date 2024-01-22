# Example of how to run the simulator via a file. 

import Pkg;

try
    using ResumableFunctions
    using ConcurrentSim
    using NativeSVG
catch
    Pkg.add("ResumableFunctions")
    Pkg.add("ConcurrentSim")
    Pkg.add(url="https://github.com/BenLauwens/NativeSVG.jl.git")
end



include("src/structs.jl")
include("src/simulator.jl")
include("src/visualizer.jl")
include("programs/SAXPY.jl")

println("===")

sim = Simulation()
arch = Architecture(1, SM(sim,32,8), GlobalMemory(), [],16);

loadSAXPY(arch) # Loads the SAXPY instructions in the kernel

@process simulate(sim, arch, verbosity=3, recordUsage=true, recordOperandData=[3, 1])
run(sim, 50000)

# visualize(arch)