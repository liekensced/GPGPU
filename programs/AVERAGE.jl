
function loadAVG(arch::Architecture, data::Vector{Int})
    # Initializing Memory

    arch.globalMemory.data[1:length(data)] = data # Vector X
    arch.SM.constantMemory.data[1] = length(data) # Length

    # Generate instructions
    function avgInstructions(idx)
        return [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("MULT", [Address(1), 2], 2), # Multiply TID*2
            Instruction("ADD", [Address(2), -1], 3), # TID*2 + 1
            Instruction("LOADG", [Address(2)], 4), # Load x[tid*2] from shared Mem
            Instruction("LOADG", [Address(3)], 5), # Load x[tid*2+1] from shared Mem
            Instruction("ADD", [Address(4),Address(5)], 6), # Calculate sum
            Instruction("WBS", [Address(1), Address(6)]) # Write the solution to shared memory at index TID
        ]
    end
    # 32 threads will be doing the same thing (SIMD)

    warpSize = arch.SM.coreAmount

    #idx NEEDS TO START @ 0
    u::Int = ceil(length(data)/(2*warpSize))-1
    
    for idx = (0:u) * warpSize # now it doesn't really matter if it's too high
        push!(arch.kernel, avgInstructions(idx)...)
    end

    loadAVGStep(arch, length(data))
end

function loadAVG(arch::Architecture)
    loadAVG(arch::Architecture, collect(1:128))
end

function loadAVGStep(arch::Architecture, vectorLength::Int)
    # Generate instructions
    function avgInstructions(idx::Int)
        return [
            Instruction("LOADTID", [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("MULT", [Address(1), 2], 2), # Multiply TID*2
            Instruction("ADD", [Address(2), -1], 3), # TID*2 + 1
            Instruction("LOADS", [Address(2)], 4), # Load x[tid*2] from shared Mem
            Instruction("LOADS", [Address(3)], 5), # Load x[tid*2+1] from shared Mem
            Instruction("ADD", [Address(4),Address(5)], 6), # Calculate sum
            Instruction("WBS", [Address(1), Address(6)]) # Write the solution to shared memory at index TID
        ]
    end
    # 32 threads will be doing the same thing (SIMD)

    warpSize = arch.SM.coreAmount

    #idx NEEDS TO START @ 0
    while vectorLength > 0
        vectorLength = div(vectorLength, 2)
        push!(arch.kernel, Instruction("SYNCTHREADS",[]))

        u::Int = ceil(vectorLength/(warpSize*2))-1
        for idx = (0:u) * warpSize # Not enough to do the 1000 but no < N check implemented yet
            push!(arch.kernel, avgInstructions(idx)...)
        end
    end
end 

function runAVG()
	sim = Simulation()
	arch = Architecture(1, SM(sim,32,1), GlobalMemory(), [],32*5);

	loadAVG(arch)
	
	push!(arch.kernel, [
				Instruction("LOADTID",[0],1),
				Instruction("JUMP.>",[Address(1), 1, 4]),
				Instruction("LOADS",[1],2),
				Instruction("LOADC",[1],3),
				Instruction("INT.div",[Address(2), Address(3)], 4),
				Instruction("WBS",[1, Address(4)])
			]...)
	@process simulate(sim, arch, verbosity=0,kernelLength=7, recordOperandData=[6,1])

	run(sim, 5000)
    println(arch.hist[end].sharedMemory)
end