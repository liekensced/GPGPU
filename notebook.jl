### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 0bf993e0-e09b-4c73-a1e8-8c25f1b7b315
import Pkg;

# ╔═╡ 1366aed0-d0ee-4c3a-827c-157a22b27975
try
    using ResumableFunctions
    using ConcurrentSim
    using NativeSVG
catch
    Pkg.add("ResumableFunctions")
    Pkg.add("ConcurrentSim")
    Pkg.add(url="https://github.com/BenLauwens/NativeSVG.jl.git")
end

# ╔═╡ a3211620-6c2d-4ca7-ad4c-06ff64a9680e
using PlutoUI

# ╔═╡ af566a66-9560-49a7-afd0-ddf1eb635381
md"First really weird pluto imports"

# ╔═╡ bae056b2-6c8f-4923-b7ad-d087340c06c6
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ ed35d405-c330-4cad-8c42-82038c78da9f
N = ingredients("pluto_exporter.jl")

# ╔═╡ 7b15b1c2-91e5-4220-948f-836d33d267c8
import .N: Architecture,InstructionRecord,SM,Instruction,CUDACore,GlobalMemory,ConstantMemory,SharedMemory, DEFAULT_MEMREQUESTSIZE,DEFAULT_LATENCY,DEFAULT_INITIATION_INTERVALS,simulate,simulateWarp,simulateThread,visualize,loadSAXPY,stats

# ╔═╡ 25b25527-e097-4cfd-9332-c5cba3d66674
md"# GP GPU"

# ╔═╡ ffe0dc36-4d74-479d-b559-b5f7dbf2f38d
html"""
<h3> Nvidia Ampere GPU </h3>
<img src="https://www.nextplatform.com/wp-content/uploads/2020/05/nvidia-ampere-ga100-block-diagram.jpg">
<h3> Streaming Multiprocessor SIMD </h3>
<img src="https://www.geeks3d.com/public/jegx/201001/fermi_gt100_sm.jpg">
<h3> CUDA Core </h3>
<img src="https://www.electronicshub.org/wp-content/uploads/2021/06/Cuda-Core-1.png">
"""

# ╔═╡ f8b63eaf-8bdd-425b-af4f-286ba59ee54b
md"""## Unit testing simulation
### Environment"""

# ╔═╡ 73b34bf5-439b-4ca2-9bb6-7ead92d31191
const SAXPY_X = collect(1:1000) .+ 5

# ╔═╡ 45f6b0a1-41e1-4446-a9d8-2af9c1d27df4
const SAXPY_Y = collect(2:2:2000)

# ╔═╡ d6439072-68e8-469c-bfd6-ab42e721f07d
const SAXPY_a = 5

# ╔═╡ d7c02444-0f3f-4182-9db1-4189d81d2293
function cleanEnv()
    sim = Simulation()

    arch = Architecture(1, SM(sim), GlobalMemory(), []);

    arch.globalMemory.data[1:1000] = SAXPY_X # Vector X
    arch.globalMemory.data[1001:2000] = SAXPY_Y # Vector Y
    arch.SM.constantMemory.data[1] = SAXPY_a # Scalar a
    return sim,arch
end

# ╔═╡ 23b24828-69d8-4565-aa68-1bb1f709400c
md"### Load TID"

# ╔═╡ 134c2e30-0ca4-41bc-82a9-89c0eb86d1ba
@resumable function testTID(sim,arch)
	insn = Instruction("LOADTID", [], [0], 1)
	@process simulateWarp(sim,arch,arch.SM,insn)
end

# ╔═╡ 8fc177eb-8db0-4bca-baf5-42edba1a7cc5
let
	## Test LOADTID
	println("=== Test LOADTID ===")
	
	sim,arch = cleanEnv();
	@process testTID(sim,arch)
	run(sim)

	@assert arch.SM.cores[1].operands[1] == 1
	@assert arch.SM.cores[32].operands[1] == 32
	
	visualize(arch)
end

# ╔═╡ 4d55e653-52e5-44f1-aad0-42403f38e9aa
md"### Load Constant"

# ╔═╡ e034b7c5-2a1c-4ac5-8a11-a2dbe821eaf6
@resumable function testLOADC(sim,arch)
	insn = Instruction("LOADC", [], [1], 2) 
	@process simulateWarp(sim,arch,arch.SM,insn)
end

# ╔═╡ 6261773d-cfb3-4235-a465-50cc9d72dfcf
let
	## Test LOADC
	println("=== Test LOADC ===")
	sim,arch = cleanEnv();
	@process testLOADC(sim,arch)
	run(sim)
	@assert arch.SM.cores[1].operands[2] == SAXPY_a # == 5
	@assert arch.SM.cores[32].operands[2] == SAXPY_a # == 5
	
	visualize(arch)
end

# ╔═╡ 89be8365-44d2-4210-9a81-56cfe908366f
md"### Load Global"

# ╔═╡ 70a3028c-c79f-4e1c-aa9a-68cfc2d11b8f
@resumable function testLOADG(sim,arch)
	insn = Instruction("LOADG", [1], [], 3)
	@process simulateWarp(sim,arch,arch.SM,insn)
end

# ╔═╡ 5ec11660-f487-4ba1-b395-0e918e44ba5f
let
	## Test LOADG relative to TID
	println("=== Test LOADG ===")
	sim,arch = cleanEnv();
	@process testTID(sim,arch)
	@process testLOADG(sim,arch)
	run(sim)
	@assert arch.SM.cores[1].operands[3] == SAXPY_X[1] # == 1
	@assert arch.SM.cores[32].operands[3] == SAXPY_X[32] # == 32
	@assert 1 in arch.SM.globalMemoryReceived # Check whether the request bundling works
	@assert 2 ∉ arch.SM.globalMemoryReceived
	
	visualize(arch)
end

# ╔═╡ f20f73cc-9228-451b-8af0-5f1836410d64
md"### Multiplication"

# ╔═╡ ef55bf74-4a2b-47c0-aaca-2395be6cbe17
@resumable function testMULT(sim,arch)
    insn = Instruction("MULT", [2, 3], [], 5)
    @process simulateWarp(sim,arch,arch.SM,insn)
end

# ╔═╡ 9a6d4843-342d-496b-9278-289f48f86d41
let
	## Test MULT relative to prevs
	println("=== Test MULT ===")

	sim,arch = cleanEnv();
	@process testTID(sim,arch)
	@process testLOADC(sim,arch)
	@process testLOADG(sim,arch)
	@process testMULT(sim,arch)

	run(sim)
	@assert arch.SM.cores[1].operands[5] == SAXPY_X[1]*5
	@assert arch.SM.cores[32].operands[5] == SAXPY_X[32]*5
	
	visualize(arch)
end

# ╔═╡ 03b2d4d2-16f1-43a0-9a22-feefc9bae208
md"### Addition"

# ╔═╡ 5e674cc6-9b41-4b24-8fb4-08b77bb201e8
@resumable function testADD(sim,arch)
    insn = Instruction("ADD", [1], [1000], 4) # Calculate address of y
    @process simulateWarp(sim,arch,arch.SM,insn)
    insn2 = Instruction("LOADG", [4], [], 6)
    @process simulateWarp(sim,arch,arch.SM,insn2)
end

# ╔═╡ 7d231bc8-7994-48dd-b1b2-f9d19c29f586
let
	## Test ADD relative to prevs and load a global var from it
	println("=== Test ADD ===")

	sim,arch = cleanEnv();
	@process testTID(sim,arch)
	@process testADD(sim,arch)

	run(sim)
	@assert arch.SM.cores[1].operands[4] == arch.SM.cores[1].operands[1] + 1000
	@assert arch.SM.cores[end].operands[4] == arch.SM.cores[end].operands[1] + 1000

	@assert arch.SM.cores[1].operands[6] == SAXPY_Y[1] # == 2 (2:2:2000)
	@assert arch.SM.cores[32].operands[6] == SAXPY_Y[32] # == 64
	
	visualize(arch)
end

# ╔═╡ 6bafd9f1-ff94-4ed7-bce9-93e53cf021c8
md"### Write back"

# ╔═╡ d859a90f-422e-4d8d-8542-d04db0394cc9

@resumable function testWBS(sim,arch)
    insn = Instruction("WBS", [1, 1], [], 0) # The wb actually doesn't get used
    @process simulateWarp(sim,arch,arch.SM,insn)
end


# ╔═╡ b37c3392-8d59-4198-9b2e-7fb9cb8859ad
let
	## Test WB to sharedMemory. It saves the TID at addres TID
	println("=== Test WBS ===")
	sim, arch = cleanEnv();
	@process testTID(sim,arch)
	@process testWBS(sim,arch)
	run(sim)

	@assert arch.SM.sharedMemory.data[1:32] == collect(1:32)
	@assert arch.SM.sharedMemory.data[33] == 0
	
	visualize(arch)
end

# ╔═╡ 1d46e2db-ed70-4d3d-b361-31ddf72153fe
md"### Test SAXPY instructions"

# ╔═╡ ccf3e184-3b0c-47c1-bce9-150bf3037a90
html"""
General CUDA imlementation:
<pre>
__global__
void saxpy(int n, float a, float * restrict x, float * restrict y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

...
int N = 1<<20;
cudaMemcpy(d_x, x, N, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y, N, cudaMemcpyHostToDevice);

// Perform SAXPY on 1M elements
saxpy<<<4096,256>>>(N, 2.0, d_x, d_y);

cudaMemcpy(y, d_y, N, cudaMemcpyDeviceToHost);
</pre>
"""

# ╔═╡ 7382de8a-2465-4ac4-893d-cc48e7df8df3
let
## Test SAXPY insn's completely for 1 loop

println("=== Test SAXPY ===")
@resumable function testSAXPY(sim,arch)
    @process testTID(sim,arch)
    @yield timeout(sim,1)
    @process testLOADC(sim,arch)
    @yield timeout(sim,1)

    @process testLOADG(sim,arch)
    @yield timeout(sim,1)

    @process testMULT(sim,arch)
    @yield timeout(sim,1)

    @process testADD(sim,arch)
    @yield timeout(sim,1)

    insn = Instruction("ADD", [5,6], [], 7)
    @process simulateWarp(sim,arch,arch.SM,insn)
    @yield timeout(sim,1)
    
    insn2 = Instruction("WBS", [1,7], [], 0) # Write the solution to shared memory at index TID
    @process simulateWarp(sim,arch,arch.SM,insn2)
    
end
sim,arch = cleanEnv();

@process testSAXPY(sim,arch)

run(sim,1000)

if (true)
    println(["tid","a","x","yaddr","a*x","y","result"])
    for i in 1:32
        print(arch.SM.cores[i].operands[1:7])
        print(" : ")
        println(SAXPY_a*SAXPY_X[i] + SAXPY_Y[i])
    end
end

for i in 1:32
    @assert arch.SM.cores[i].operands[7] == SAXPY_a*SAXPY_X[i] + SAXPY_Y[i]
    @assert arch.SM.sharedMemory.data[i] == SAXPY_a*SAXPY_X[i] + SAXPY_Y[i]
end
	visualize(arch)
end

# ╔═╡ bc3e1e57-910f-4bc0-910c-7211e4c2efbc
md"""
## Simulating SAXPY
### First attempt
"""

# ╔═╡ cbf92b36-797e-421c-b4e0-66225822b250
function loadSAXPYAttempt(arch::Architecture, saxpyInstructionsAttempt::Function)
    # Initializing Memory

    arch.globalMemory.data[1:1000] = collect(1:1000) # Vector X
    arch.globalMemory.data[1001:2000] = collect(2:2:2000) # Vector Y
    arch.SM.constantMemory.data[1] = 5 # Scalar a

    # Generate instructions
    
    # 32 threads will be doing the same thing (SIMD)

    warpSize = arch.SM.coreAmount

    #idx NEEDS TO START @ 0
    for idx = (0:7) * warpSize # Not enough to do the 1000 but no < N check implemented yet
        push!(arch.kernel, saxpyInstructionsAttempt(idx)...)
    end
end

# ╔═╡ bd1a592d-8696-4ff8-82ff-2b75d7b68ed2
let
	sim = Simulation()
	arch = Architecture(1, SM(sim), GlobalMemory(), []);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
            Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
            Instruction("ADD", [1], [1000], 4), # Calculate address of y
            Instruction("LOADG", [4], [], 3), # Load y[tid] from global Mem
            Instruction("ADD", [2, 3], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 0) # Write the solution to shared memory at index TID
        ])

	@process simulate(sim, arch)
	run(sim, 50000)
	
	visualize(arch)
end

# ╔═╡ ee62c206-90c3-4119-aa5b-7910a94fad69
md"While this does optimize the amount of registers used (only 4), it creates a bottleneck at the fetching of the global data. -> Save the second LOADG in another register"

# ╔═╡ 2e75337b-2913-4e42-9468-cb1764176505
let
	sim = Simulation()
	arch = Architecture(1, SM(sim), GlobalMemory(), []);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
            Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
            Instruction("ADD", [1], [1000], 4), # Calculate address of y
            Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 0) # Write the solution to shared memory at index TID
        ])

	@process simulate(sim, arch)
	run(sim, 50000)
	
	visualize(arch)
end

# ╔═╡ 71f0e296-91b9-4715-9b3d-5f85ea51be5b
md"The biggest bottleneck is still fetching the data from global memory. If we could theoretically up the MEMREQUESTSIZE it could get a lot faster. (The time to go to global memory is pretty much fixed.)"

# ╔═╡ 72c485b6-a713-4cee-b22b-8228a5771fc2
let
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,2), GlobalMemory(), [], 64 + 16);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
			Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
			Instruction("ADD", [1], [1000], 4), # Calculate address of y
			Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 0) # Write the solution to shared memory at index TID
        ])
	@process simulate(sim, arch)
	run(sim, 50000)
	
	print(stats(arch))
	
	visualize(arch)
end

# ╔═╡ 226214a6-4f95-40c2-8763-591db5051b5b
md"Here we added 16 to the default memRequestSize (64). This illustrates the fact that the warp waits untill the last thread is finished. At the third warp, 16 of the 32 threads have their data in cache, the other 16 have to wait which stalls the entire warp. The 4th time enough data is in cache so we win a call to global memory. This will be the case every 4th warp."

# ╔═╡ 54c2f407-3f4f-4a54-b06d-d9dfa262ceab


# ╔═╡ Cell order:
# ╟─af566a66-9560-49a7-afd0-ddf1eb635381
# ╠═a3211620-6c2d-4ca7-ad4c-06ff64a9680e
# ╠═0bf993e0-e09b-4c73-a1e8-8c25f1b7b315
# ╠═1366aed0-d0ee-4c3a-827c-157a22b27975
# ╟─bae056b2-6c8f-4923-b7ad-d087340c06c6
# ╠═ed35d405-c330-4cad-8c42-82038c78da9f
# ╠═7b15b1c2-91e5-4220-948f-836d33d267c8
# ╟─25b25527-e097-4cfd-9332-c5cba3d66674
# ╠═ffe0dc36-4d74-479d-b559-b5f7dbf2f38d
# ╟─f8b63eaf-8bdd-425b-af4f-286ba59ee54b
# ╠═73b34bf5-439b-4ca2-9bb6-7ead92d31191
# ╠═45f6b0a1-41e1-4446-a9d8-2af9c1d27df4
# ╠═d6439072-68e8-469c-bfd6-ab42e721f07d
# ╠═d7c02444-0f3f-4182-9db1-4189d81d2293
# ╟─23b24828-69d8-4565-aa68-1bb1f709400c
# ╠═134c2e30-0ca4-41bc-82a9-89c0eb86d1ba
# ╠═8fc177eb-8db0-4bca-baf5-42edba1a7cc5
# ╟─4d55e653-52e5-44f1-aad0-42403f38e9aa
# ╠═e034b7c5-2a1c-4ac5-8a11-a2dbe821eaf6
# ╠═6261773d-cfb3-4235-a465-50cc9d72dfcf
# ╠═89be8365-44d2-4210-9a81-56cfe908366f
# ╠═70a3028c-c79f-4e1c-aa9a-68cfc2d11b8f
# ╠═5ec11660-f487-4ba1-b395-0e918e44ba5f
# ╟─f20f73cc-9228-451b-8af0-5f1836410d64
# ╠═ef55bf74-4a2b-47c0-aaca-2395be6cbe17
# ╠═9a6d4843-342d-496b-9278-289f48f86d41
# ╟─03b2d4d2-16f1-43a0-9a22-feefc9bae208
# ╠═5e674cc6-9b41-4b24-8fb4-08b77bb201e8
# ╠═7d231bc8-7994-48dd-b1b2-f9d19c29f586
# ╟─6bafd9f1-ff94-4ed7-bce9-93e53cf021c8
# ╠═d859a90f-422e-4d8d-8542-d04db0394cc9
# ╠═b37c3392-8d59-4198-9b2e-7fb9cb8859ad
# ╟─1d46e2db-ed70-4d3d-b361-31ddf72153fe
# ╟─ccf3e184-3b0c-47c1-bce9-150bf3037a90
# ╠═7382de8a-2465-4ac4-893d-cc48e7df8df3
# ╟─bc3e1e57-910f-4bc0-910c-7211e4c2efbc
# ╠═cbf92b36-797e-421c-b4e0-66225822b250
# ╠═bd1a592d-8696-4ff8-82ff-2b75d7b68ed2
# ╟─ee62c206-90c3-4119-aa5b-7910a94fad69
# ╠═2e75337b-2913-4e42-9468-cb1764176505
# ╟─71f0e296-91b9-4715-9b3d-5f85ea51be5b
# ╠═72c485b6-a713-4cee-b22b-8228a5771fc2
# ╟─226214a6-4f95-40c2-8763-591db5051b5b
# ╠═54c2f407-3f4f-4a54-b06d-d9dfa262ceab
