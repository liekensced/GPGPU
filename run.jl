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
arch = Architecture(1, SM(sim,32,8), GlobalMemory(), [],32*5);

loadSAXPY(arch)

@process simulate(sim, arch, 8)
run(sim, 50000)

# visualize(arch)