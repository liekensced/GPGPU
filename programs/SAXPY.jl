# SAXPY
# 
# int tid = blockIdx.x * blockDim.x + threadIdx.x; # Each thread has acces to it's tid
#
# if (tid < n) {
#     y[tid] = a * x[tid] + y[tid];
# }

function loadSAXPY(arch::Architecture)
    # Initializing Memory

    arch.globalMemory.data[1:1000] = collect(1:1000) # Vector X
    arch.globalMemory.data[1001:2000] = collect(2:2:2000) # Vector Y
    arch.SM.constantMemory.data[1] = 5 # Scalar a

    # Generate instructions
    function saxpyInstructions(idx)
        return [
            Instruction("LOADTID", [], [idx], 1), # TID = blockIdx.x * blockDim.x + threadIdx.x; # Thread ID is known via hardware, blockIdx and blockDim should be given
            Instruction("LOADC", [], [1], 2), # Load a from constant Mem
            Instruction("LOADG", [1], [], 3), # Load x[tid] from global Mem
            Instruction("MULT", [2, 3], [], 2), # Multiply a*x[tid]
            Instruction("ADD", [1], [1000], 4), # Calculate address of y
            Instruction("LOADG", [4], [], 5), # Load y[tid] from global Mem
            Instruction("ADD", [2, 5], [], 2), # Save solution in 2
            Instruction("WBS", [1, 2], [], 1) # Write the solution to shared memory at index TID
        ]
    end
    # 32 threads will be doing the same thing (SIMD)

    warpSize = arch.SM.coreAmount

    #idx NEEDS TO START @ 0
    for idx = (0:5) * warpSize # Not enough to do the 1000 but no < N check implemented yet
        push!(arch.kernel, saxpyInstructions(idx)...)
    end
end