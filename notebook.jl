### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 0bf993e0-e09b-4c73-a1e8-8c25f1b7b315
import Pkg;

# ╔═╡ 1366aed0-d0ee-4c3a-827c-157a22b27975
try
    using ResumableFunctions
    using ConcurrentSim
    using NativeSVG
	using PlutoUI
	using PrettyTables
catch
    Pkg.add("ResumableFunctions")
    Pkg.add("ConcurrentSim")
    Pkg.add(url="https://github.com/BenLauwens/NativeSVG.jl.git")
	Pkg.add("PlutoUI")
	Pkg.add("PrettyTables")
end

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
import .N: Architecture,InstructionRecord,SM,Instruction,CUDACore,GlobalMemory,ConstantMemory,SharedMemory,DEFAULT_MEMREQUESTSIZE,DEFAULT_LATENCY,DEFAULT_INITIATION_INTERVALS,simulate,simulateWarp,simulateThread,visualize,loadSAXPY,stats, visualizeData, Address, loadAVG

# ╔═╡ 83ac706a-fec9-437c-91e1-d11da40eeb08
TableOfContents()

# ╔═╡ 25b25527-e097-4cfd-9332-c5cba3d66674
md"# GP GPU"

# ╔═╡ 9e335840-be37-4c9b-ad9d-6ae36ef5f239
md"*General Purpose GPU*"

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

# ╔═╡ fe1d66d1-eff7-46e2-94bd-c7fdd83b765a
md"R = aX+Y"

# ╔═╡ f3a6adad-ff90-4b67-beee-74844087cc1f
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

# ╔═╡ d5b566de-bca2-400b-b171-2af830715be3
html"""Loads the TID in the wb adress.<br />
<code> valsOperands[1] </code> is the <code> blockIdx.x * blockDim.x</code> value from the kernel/warp scheduler. 
"""

# ╔═╡ 134c2e30-0ca4-41bc-82a9-89c0eb86d1ba
@resumable function testTID(sim,arch)
	insn = Instruction("LOADTID", [], [0], 1)
	@process simulateWarp(sim,arch,arch.SM,insn,4)
end

# ╔═╡ 8fc177eb-8db0-4bca-baf5-42edba1a7cc5
let
	## Test LOADTID
	println("=== Test LOADTID ===")
	
	sim,arch = cleanEnv();
	@process testTID(sim,arch)
	run(sim)

	@assert arch.SM.cores[1].operands[1][1] == 1
	@assert arch.SM.cores[32].operands[1][1] == 32
	
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
	@assert arch.SM.cores[1].operands[1][2] == SAXPY_a # == 5
	@assert arch.SM.cores[32].operands[1][2] == SAXPY_a # == 5
	
	visualize(arch)
end

# ╔═╡ 89be8365-44d2-4210-9a81-56cfe908366f
md"### Load Global"

# ╔═╡ 70a3028c-c79f-4e1c-aa9a-68cfc2d11b8f
@resumable function testLOADG(sim,arch)
	insn = Instruction("LOADG", [Address(1)], 3)
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
	@assert arch.SM.cores[1].operands[1][3] == SAXPY_X[1] # == 1
	@assert arch.SM.cores[32].operands[1][3] == SAXPY_X[32] # == 32
	@assert 1 in arch.SM.globalMemoryReceived # Check whether the request is bundling works
	@assert 2 ∉ arch.SM.globalMemoryReceived
	
	visualize(arch)
end

# ╔═╡ 99002efe-9fee-42f3-8fab-b62503a90120
md"The request can only begin after the TID is fetched."

# ╔═╡ 16dcbab1-d604-4bff-87ff-570caa504141
md"Only the first thread issues a request to globalMem which includes the data of the other threads."

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
	@assert arch.SM.cores[1].operands[1][5] == SAXPY_X[1]*5
	@assert arch.SM.cores[32].operands[1][5] == SAXPY_X[32]*5
	
	visualize(arch)
end

# ╔═╡ 49986231-132f-4cf2-a8b5-0b4a492aed87
md"The product can only begin after the constant and X val is loaded. The global read is the bottleneck."

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
	@assert arch.SM.cores[1].operands[1][4] == arch.SM.cores[1].operands[1][1] + 1000
	@assert arch.SM.cores[end].operands[1][4] == arch.SM.cores[end].operands[1][1] + 1000

	@assert arch.SM.cores[1].operands[1][6] == SAXPY_Y[1] # == 2 (2:2:2000)
	@assert arch.SM.cores[32].operands[1][6] == SAXPY_Y[32] # == 64
	
	visualize(arch)
end

# ╔═╡ ee49958c-a9cb-4e73-b199-379e89f6347f
md"The address of the Y vector is calculated. This needs to be done before it can go to global memory."

# ╔═╡ 6bafd9f1-ff94-4ed7-bce9-93e53cf021c8
md"### Write back"

# ╔═╡ d859a90f-422e-4d8d-8542-d04db0394cc9

@resumable function testWBS(sim,arch)
    insn = Instruction("WBS", [1, 1], [], 0) 
	# The last value (operand wb) actually doesn't get used. It is the first given operand that determines the address in sharedMem
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

# ╔═╡ c5a290d0-d020-4451-a82c-b33bfcad8ad9
md"Using WB the CUDA core can save data from its operands to the shared memory which can be acces by other cores."

# ╔═╡ 1d46e2db-ed70-4d3d-b361-31ddf72153fe
md"### Test SAXPY instructions"

# ╔═╡ 2c6b312c-5f0a-4761-a2f5-8c5eb4ac52c5
md"S=a*X+Y"

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
    @process testLOADC(sim,arch)

    @process testLOADG(sim,arch)

    @process testMULT(sim,arch)

    @process testADD(sim,arch)

    insn = Instruction("ADD", [5,6], [], 7)
    @process simulateWarp(sim,arch,arch.SM,insn)
    
    insn2 = Instruction("WBS", [1,7], [], 0) # Write the solution to shared memory at index TID
    @process simulateWarp(sim,arch,arch.SM,insn2)
    
end
sim,arch = cleanEnv();

@process testSAXPY(sim,arch)

run(sim,1000)

if (true)
    println(["tid","a","x","yaddr","a*x","y","result"])
    for i in 1:32
        print(arch.SM.cores[i].operands[1][1:7])
        print(" : ")
        println(SAXPY_a*SAXPY_X[i] + SAXPY_Y[i])
    end
end

for i in 1:32
    @assert arch.SM.cores[i].operands[1][7] == SAXPY_a*SAXPY_X[i] + SAXPY_Y[i]
    @assert arch.SM.sharedMemory.data[i] == SAXPY_a*SAXPY_X[i] + SAXPY_Y[i]
end
	visualize(arch)
end

# ╔═╡ fbc01d77-125d-420a-959e-aed454362e1e
md"This tests all the instruction. Note we simulate only the warps, not the SM (with it's warp schedulars etc)."

# ╔═╡ bc3e1e57-910f-4bc0-910c-7211e4c2efbc
md"""
## Simulating SAXPY
### First attempt
"""

# ╔═╡ cbf92b36-797e-421c-b4e0-66225822b250
function loadSAXPYAttempt(arch::Architecture, saxpyInstructionsAttempt::Function,blockAmount=7)
    # Initializing Memory

    arch.globalMemory.data[1:1000] = SAXPY_X # Vector X
    arch.globalMemory.data[1001:2000] = SAXPY_Y # Vector Y
    arch.SM.constantMemory.data[1] = 5 # Scalar a

    # Generate instructions
    
    # 32 threads will be doing the same thing (SIMD)

    warpSize = arch.SM.coreAmount

    #idx NEEDS TO START @ 0
    for idx = (0:blockAmount) * 8 # Not enough to do the 1000 but no < N check implemented yet
        push!(arch.kernel, saxpyInstructionsAttempt(idx)...)
    end
end

# ╔═╡ 5e8e420b-9352-44ca-a121-b1b7dbbd1872
loaded_archs= Vector()

# ╔═╡ bd1a592d-8696-4ff8-82ff-2b75d7b68ed2
let
	sim_1 = Simulation()
	arch_1 = Architecture(1, SM(sim_1), GlobalMemory(), []);
	

	loadSAXPYAttempt(arch_1, idx ->
		[
			Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
			Instruction("LOADC", [1], 2), # Load a from constant Mem
			Instruction("LOADG", [Address(1)], 3), # Load x[tid] from global Mem
			Instruction("MULT", [Address(2), Address(3)], 2), # Multiply a*x[tid]
			Instruction("ADD", [Address(1),1000], 4), # Calculate address of y
			Instruction("LOADG", [Address(4)], 3), # Load y[tid] from global Mem
			Instruction("ADD", [Address(2), Address(3)], 2), # Save solution in 2
			Instruction("WBS", [Address(1), Address(2)], 0) # Write the solution to shared memory at index TID
			])

	@process simulate(sim_1, arch_1, recordOperandData=[4,1])
	push!(loaded_archs, arch_1)
	run(sim_1, 50000)
end

# ╔═╡ 389208d2-2503-4212-88ad-9d3408ba986f
@bind currentArch Select(string.(1:length(loaded_archs)))

# ╔═╡ e5720132-1dd8-4679-8956-b61d89f9f97f
@bind currentWarp Slider(1:length(loaded_archs[parse(Int, currentArch)].hist), default=1,show_value=true)

# ╔═╡ 27253221-2742-46bc-916b-5025203f556f
visualize(loaded_archs[parse(Int, currentArch)],false, currentWarp)

# ╔═╡ 9f3ac487-7e99-4bb6-a601-3d0b9ebb65ef
visualizeData(loaded_archs[parse(Int, currentArch)], currentWarp, sharedMemoryLines = 1)

# ╔═╡ 801b7798-1cef-4a0f-bd21-dab310bfb146
md" => 313 cycles"

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

# ╔═╡ 0c74bfdd-912c-4d53-be30-ac9a03458854
md"223 cycles (313, -28.8%)"

# ╔═╡ 71f0e296-91b9-4715-9b3d-5f85ea51be5b
md"The biggest bottleneck is still fetching the data from global memory. If we could theoretically up the MEMREQUESTSIZE it could get a lot faster. (The time to go to global memory is pretty much fixed.)"

# ╔═╡ 72c485b6-a713-4cee-b22b-8228a5771fc2
let
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,1), GlobalMemory(), [], 64 + 16);

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

# ╔═╡ 226214a6-4f95-40c2-8763-591db5051b5b
md"Here we added 16 to the default memRequestSize (64). This illustrates the fact that the warp waits untill the last thread is finished. At the third warp, 16 of the 32 threads have their data in cache, the other 16 have to wait which stalls the entire warp. The 4th time enough data is in cache so we win a call to global memory. This will be the case every 4th warp."

# ╔═╡ 53db9cca-9235-4a9b-857d-864d3e2c2a0c
let
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,1), GlobalMemory(), [], 64*50);

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

# ╔═╡ d247a670-5e09-433e-954f-5b19bd17add8
md"In reality the memRequestSize would be a lot bigger, if we set the the memRequestSize high enough that everything gets loaded, we have 176 cycles (223, -21%)"

# ╔═╡ b463a64d-7c1f-4339-bf60-12e312e18b63
md"""### Context switching
One big advantage of GPU's is zero-heap context switching. This is possible because the operands are actually saved in a register that can hold multiple contexts. The amount is dependant on the size of each context (registerSize/contextSize). So while the first warp is still fetching data, the second warp can already start executing and use the GPU idle resources.
"""

# ╔═╡ 2986df9a-e63f-45f6-9114-d6d2b0eb72ac
let
	
	CONTEXTSAMOUNT = 5 
	# instead of 1 == Total registery size / (cores * contextSize per core) 
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,CONTEXTSAMOUNT), GlobalMemory(), [], 32*5);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
			Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
			Instruction("ADD", [1], [1000], 4), # Calculate address of y
			Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 1) # Write the solution to shared memory at index TID
        ])
	@process simulate(sim, arch)
	run(sim, 50000)
	
	
	visualize(arch)
end

# ╔═╡ 93349cb7-c8c5-45ad-964d-f9a824359c47
md"We now have 104 cycles (176, -40.1%)"

# ╔═╡ 22382997-f3c1-41e9-9b7b-6284a8bc6aef
md"## Searching te limit"

# ╔═╡ 46002f99-b3e2-408c-b0d4-66cf000d151f
md"There is still a lockstep point. Let's increase CONTEXTAMOUNT so it doesn't block anymore."

# ╔═╡ 3fbcf66a-beba-46bc-ab4d-0e3786e3e987
let
	
	CONTEXTSAMOUNT = 999 # instead of 1
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,CONTEXTSAMOUNT), GlobalMemory(), [], 32*5);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
			Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
			Instruction("ADD", [1], [1000], 4), # Calculate address of y
			Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 1) # Write the solution to shared memory at index TID
        ])
	@process simulate(sim, arch)
	run(sim, 50000)

	
	visualize(arch)
end

# ╔═╡ fb4b2c7d-8d6e-422c-bbb7-53659cd5f1e7
md"We now have 81 cylces (104, -22.1%). Note that this isn't realistic"

# ╔═╡ 4bd94b2d-748f-46a7-b04d-da90f4437a42
md"The warp scheduling seems to take a lot of time so let's use zero latency scheduling" 

# ╔═╡ c2ef4a3f-8d9f-4fe0-ad95-c8db8e410974
let
	
	CONTEXTSAMOUNT = 999 # instead of 1
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,CONTEXTSAMOUNT), GlobalMemory(), [], 32*5);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
			Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
			Instruction("ADD", [1], [1000], 4), # Calculate address of y
			Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 1) # Write the solution to shared memory at index TID
        ])
	@process simulate(sim, arch,timeoutLength=0)
	run(sim, 50000)
	
	visualize(arch)
end

# ╔═╡ 70d083af-aa4a-40b4-9c0d-4bab7465f0a3
md"We now have 79 cylces (81, -2.5%), so it wasn't actually a big bottleneck."

# ╔═╡ 9afb0d78-2f50-42f2-bf66-f78830f917c9
md"Now the biggest wait is the time to go to global memory. But we only run 7 warps which normally would be much more? So we assume it is already loaded (because most of the time it would be) and set the latency to 0."

# ╔═╡ 0bf7605a-20a2-4a73-95b7-fb1263ccdf1b
let
	
	CONTEXTSAMOUNT = 999 # instead of 1
	
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,CONTEXTSAMOUNT), GlobalMemory(), [], 32*5);

	loadSAXPYAttempt(arch, idx ->
        [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
			Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
			Instruction("ADD", [1], [1000], 4), # Calculate address of y
			Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 1) # Write the solution to shared memory at index TID
        ])
	
	arch.latency["LOADG"] = 0
	
	@process simulate(sim, arch,timeoutLength=0)
	run(sim, 50000)
	
	println(["tid","a","x","yaddr","a*x","y","result"])
    for i in 1:32
        print(arch.SM.cores[i].operands[1][1:7])
        print(" : ")
        println(5*i+2*i)
    end
	arch.latency["LOADG"] = 50
	visualize(arch)
end

# ╔═╡ 8e2f91e2-22b2-44d3-a050-8e4a7fa2a552
md"Now we have 49 cycles (79, -38%)."

# ╔═╡ 85765fdd-65ad-417b-89cf-27058ac67ca9
7 * 32 * 8 * 1 # 7*32 threads, 8 insns, avg insn time

# ╔═╡ 00b812bf-0c71-4036-98e9-ab1a711980ab
md"Theory without dependencies and Resource constraints: 7 warps * max(initiationIntervals) + max(latencies):"

# ╔═╡ 1b2aee25-2614-4cc5-9a1a-40d186df93c3
7*4+5

# ╔═╡ 1f1a44ed-7e34-421e-b000-03a9187f9d07
md"# Average"

# ╔═╡ fcfff7bf-3e8b-4f9b-a4ef-68a6f5e049e9
html"See source code for more information. <code>JUMP.expr</code> gets evaluated at runtime so could be any valid Julia <code>expr</code>. Same with <code>INT.expr</code>. in terms of latencies etc they get handled as if it were \"ADD\" instructions."

# ╔═╡ c99e0eeb-2633-4da7-a02d-5ff5a864aef2
md"To show multiple contexts (if context switching is enabled) you need to change the recordOperandData = [first N operands, first N Contexts]"

# ╔═╡ e9d2ad7b-f94d-48fd-861b-c606f52d2107
let
	CONTEXTAMOUNT = 1
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,CONTEXTAMOUNT), GlobalMemory(), [],32*5);
	
	data = collect(1:200).+2
	loadAVG(arch, data)
	
	push!(arch.kernel, [
				Instruction("LOADTID",[0],1),
				Instruction("JUMP.>",[Address(1), 1, 4]),
				Instruction("LOADS",[1],2),
				Instruction("LOADC",[1],3),
				Instruction("INT.div",[Address(2), Address(3)], 4),
				Instruction("WBS",[1, Address(4)])
			]...)
	
	@process simulate(sim, arch, verbosity=4,kernelLength=7, recordOperandData=[6,1])

	run(sim, 5000)
	push!(loaded_archs,arch)
	(sum(data), sum(data)/length(data))
end

# ╔═╡ 4b9347cb-da0e-4358-95ee-7e41fafc5949
@bind currentWarp2 Slider(1:length(loaded_archs[end].hist), default=1,show_value=true)

# ╔═╡ a9cd4bf9-2640-4f36-aaad-96875f99358a
visualize(loaded_archs[end],false, currentWarp2)

# ╔═╡ 35476b13-3365-4543-aa33-901f1e11ed22
visualizeData(loaded_archs[end], currentWarp2, sharedMemoryLines=9)

# ╔═╡ 289aec33-b8f7-4577-a387-8b1f8a0220fb
loaded_archs[end].hist[end].endTime

# ╔═╡ Cell order:
# ╟─af566a66-9560-49a7-afd0-ddf1eb635381
# ╠═0bf993e0-e09b-4c73-a1e8-8c25f1b7b315
# ╠═1366aed0-d0ee-4c3a-827c-157a22b27975
# ╟─bae056b2-6c8f-4923-b7ad-d087340c06c6
# ╠═ed35d405-c330-4cad-8c42-82038c78da9f
# ╠═7b15b1c2-91e5-4220-948f-836d33d267c8
# ╠═83ac706a-fec9-437c-91e1-d11da40eeb08
# ╟─25b25527-e097-4cfd-9332-c5cba3d66674
# ╟─9e335840-be37-4c9b-ad9d-6ae36ef5f239
# ╠═ffe0dc36-4d74-479d-b559-b5f7dbf2f38d
# ╟─f8b63eaf-8bdd-425b-af4f-286ba59ee54b
# ╟─fe1d66d1-eff7-46e2-94bd-c7fdd83b765a
# ╟─f3a6adad-ff90-4b67-beee-74844087cc1f
# ╠═73b34bf5-439b-4ca2-9bb6-7ead92d31191
# ╠═45f6b0a1-41e1-4446-a9d8-2af9c1d27df4
# ╠═d6439072-68e8-469c-bfd6-ab42e721f07d
# ╠═d7c02444-0f3f-4182-9db1-4189d81d2293
# ╟─23b24828-69d8-4565-aa68-1bb1f709400c
# ╟─d5b566de-bca2-400b-b171-2af830715be3
# ╠═134c2e30-0ca4-41bc-82a9-89c0eb86d1ba
# ╠═8fc177eb-8db0-4bca-baf5-42edba1a7cc5
# ╟─4d55e653-52e5-44f1-aad0-42403f38e9aa
# ╠═e034b7c5-2a1c-4ac5-8a11-a2dbe821eaf6
# ╠═6261773d-cfb3-4235-a465-50cc9d72dfcf
# ╟─89be8365-44d2-4210-9a81-56cfe908366f
# ╠═70a3028c-c79f-4e1c-aa9a-68cfc2d11b8f
# ╠═5ec11660-f487-4ba1-b395-0e918e44ba5f
# ╟─99002efe-9fee-42f3-8fab-b62503a90120
# ╟─16dcbab1-d604-4bff-87ff-570caa504141
# ╟─f20f73cc-9228-451b-8af0-5f1836410d64
# ╠═ef55bf74-4a2b-47c0-aaca-2395be6cbe17
# ╠═9a6d4843-342d-496b-9278-289f48f86d41
# ╟─49986231-132f-4cf2-a8b5-0b4a492aed87
# ╟─03b2d4d2-16f1-43a0-9a22-feefc9bae208
# ╠═5e674cc6-9b41-4b24-8fb4-08b77bb201e8
# ╠═7d231bc8-7994-48dd-b1b2-f9d19c29f586
# ╟─ee49958c-a9cb-4e73-b199-379e89f6347f
# ╟─6bafd9f1-ff94-4ed7-bce9-93e53cf021c8
# ╠═d859a90f-422e-4d8d-8542-d04db0394cc9
# ╠═b37c3392-8d59-4198-9b2e-7fb9cb8859ad
# ╟─c5a290d0-d020-4451-a82c-b33bfcad8ad9
# ╟─1d46e2db-ed70-4d3d-b361-31ddf72153fe
# ╟─2c6b312c-5f0a-4761-a2f5-8c5eb4ac52c5
# ╟─ccf3e184-3b0c-47c1-bce9-150bf3037a90
# ╠═7382de8a-2465-4ac4-893d-cc48e7df8df3
# ╟─fbc01d77-125d-420a-959e-aed454362e1e
# ╟─bc3e1e57-910f-4bc0-910c-7211e4c2efbc
# ╟─cbf92b36-797e-421c-b4e0-66225822b250
# ╠═5e8e420b-9352-44ca-a121-b1b7dbbd1872
# ╠═bd1a592d-8696-4ff8-82ff-2b75d7b68ed2
# ╠═389208d2-2503-4212-88ad-9d3408ba986f
# ╟─27253221-2742-46bc-916b-5025203f556f
# ╟─e5720132-1dd8-4679-8956-b61d89f9f97f
# ╠═9f3ac487-7e99-4bb6-a601-3d0b9ebb65ef
# ╟─801b7798-1cef-4a0f-bd21-dab310bfb146
# ╟─ee62c206-90c3-4119-aa5b-7910a94fad69
# ╠═2e75337b-2913-4e42-9468-cb1764176505
# ╟─0c74bfdd-912c-4d53-be30-ac9a03458854
# ╟─71f0e296-91b9-4715-9b3d-5f85ea51be5b
# ╠═72c485b6-a713-4cee-b22b-8228a5771fc2
# ╟─226214a6-4f95-40c2-8763-591db5051b5b
# ╟─53db9cca-9235-4a9b-857d-864d3e2c2a0c
# ╟─d247a670-5e09-433e-954f-5b19bd17add8
# ╟─b463a64d-7c1f-4339-bf60-12e312e18b63
# ╠═2986df9a-e63f-45f6-9114-d6d2b0eb72ac
# ╟─93349cb7-c8c5-45ad-964d-f9a824359c47
# ╟─22382997-f3c1-41e9-9b7b-6284a8bc6aef
# ╟─46002f99-b3e2-408c-b0d4-66cf000d151f
# ╠═3fbcf66a-beba-46bc-ab4d-0e3786e3e987
# ╟─fb4b2c7d-8d6e-422c-bbb7-53659cd5f1e7
# ╟─4bd94b2d-748f-46a7-b04d-da90f4437a42
# ╟─c2ef4a3f-8d9f-4fe0-ad95-c8db8e410974
# ╟─70d083af-aa4a-40b4-9c0d-4bab7465f0a3
# ╟─9afb0d78-2f50-42f2-bf66-f78830f917c9
# ╠═0bf7605a-20a2-4a73-95b7-fb1263ccdf1b
# ╟─8e2f91e2-22b2-44d3-a050-8e4a7fa2a552
# ╠═85765fdd-65ad-417b-89cf-27058ac67ca9
# ╟─00b812bf-0c71-4036-98e9-ab1a711980ab
# ╟─1b2aee25-2614-4cc5-9a1a-40d186df93c3
# ╟─1f1a44ed-7e34-421e-b000-03a9187f9d07
# ╟─fcfff7bf-3e8b-4f9b-a4ef-68a6f5e049e9
# ╟─c99e0eeb-2633-4da7-a02d-5ff5a864aef2
# ╠═e9d2ad7b-f94d-48fd-861b-c606f52d2107
# ╟─a9cd4bf9-2640-4f36-aaad-96875f99358a
# ╠═4b9347cb-da0e-4358-95ee-7e41fafc5949
# ╟─35476b13-3365-4543-aa33-901f1e11ed22
# ╠═289aec33-b8f7-4577-a387-8b1f8a0220fb
