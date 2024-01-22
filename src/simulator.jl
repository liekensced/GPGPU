using ConcurrentSim
using ResumableFunctions

# include("structs.jl")

"""
Will record every cycle the usage of the Resources untill arch.running is false
"""
@resumable function recordingUsage(sim::Simulation, arch::Architecture)
    while arch.running
        # insnBW::Resource
        # loadwriteBW::Resource
        # intFUBW::Resource
        # globalMemoryRequestBW::Resource
        push!(arch.usage, [arch.SM.insnBW.level / arch.SM.insnBW.capacity, arch.SM.loadwriteBW.level / arch.SM.loadwriteBW.capacity, arch.SM.intFUBW.level / arch.SM.intFUBW.capacity, arch.SM.globalMemoryRequestBW.level / arch.SM.globalMemoryRequestBW.capacity])
        @yield timeout(sim, 1)
    end
end

"""
Main simulating function
"""
@resumable function simulate(sim::Simulation, arch::Architecture; kernelLength::Int=8, timeoutLength::Int=1, verbosity=1, recordUsage=true, recordOperandData=[0, 0])

    if (recordUsage)
        @process recordingUsage(sim, arch)
    end

    lockstepSize = length(arch.SM.cores[1].operands) * kernelLength # = threadAmount * kernelLength

    # First the warp scheduler acts every cycle:
    scheduledWarps = 0
    i=0
    lasti=1
    for warp in arch.kernel
        i+=1
        if warp.type == "SYNCTHREADS"
            while sum(map(x->x.threads,arch.kernel[lasti:(i-1)])) > 0
                @yield timeout(sim, 1)
            end
            lasti = i
            warp.threads = 0
        else
            @yield request(arch.SM.insnBW)
            @yield timeout(sim, timeoutLength) # Timeout for the warp to be scheduled.
            if (scheduledWarps == lockstepSize) # The 8 could be enhanced to blockSize. Assures lockstep.
                while length(arch.SM.warpBuffer) > 0
                    @yield timeout(sim, 1)
                end
                scheduledWarps = 0
            end
            scheduledWarps += 1

            @yield unlock(arch.SM.insnBW)
            warp.registerPointer = ceil(scheduledWarps / kernelLength)
            @process simulateWarp(sim, arch, arch.SM, warp, verbosity, recordOperandData) # Yielding insnBW implies a free warp in this case.
            # intentionally not yielded except if Jump

            if length(warp.type) > 5 && warp.type[1:5] == "JUMP."
                while (warp.threads) > 0 # Await result of JUMP's 
                    @yield timeout(sim, 1)
                end
            end
        end
    end

    while sum(map(x->x.threads,arch.kernel[lasti:end])) > 0
        @yield timeout(sim, 1)
    end

    arch.running = false
end

@resumable function simulateWarp(sim::Simulation, arch, sm::SM, warp::Instruction, verbosity=1, recordOperandData=[0, 0])
    push!(sm.warpBuffer, warp)
    startTime = now(sim)
    for threadID in 1:sm.coreAmount
        @process simulateThread(sim, arch, sm, warp, threadID,verbosity)
    end
    

    @yield timeout(sim, 0)
    while warp.threads > 0
        @yield timeout(sim, 1)
    end

    verbosity >= 1 && println("Executed warp: \e[32m $(warp.type)\e[0m @ $(now(sim)) \e[90m")
    verbosity >= 4 && println(arch.SM.cores[1].operands[1][1:7])
    verbosity >= 2 && println(arch.SM.cores[end].operands[1][1:7])
    verbosity >= 3 && println([insn.type for insn in arch.SM.warpBuffer])
    print("\e[0m")

    operandData = []
    for iOperand in 1:recordOperandData[1]
        for iContext in 1:recordOperandData[2]
            push!(operandData, map(x -> x.operands[iContext][iOperand], arch.SM.cores))
        end
    end
    push!(arch.hist, InstructionRecord(startTime, now(sim), warp, operandData, arch.SM.sharedMemory.data[1:300]))
    deleteat!(sm.warpBuffer, findfirst(x -> x == warp, sm.warpBuffer))
end

@resumable function simulateThread(sim::Simulation, arch, sm::SM, warp::Instruction, threadID,verbosity=0)

    # Check if current thread is masked (for branching)
    maskedThreads = []
    if sm.cores[threadID].mask > 0
        verbosity > 0 && push!(maskedThreads, threadID)
        sm.cores[threadID].mask -= 1
        warp.threads -= 1
        return
    end
    !isempty(maskedThreads) && println("$(warp.type) is masked at #$maskedThreads")

    MEMREQUESTSIZE = arch.memRequestSize
    LATENCY = arch.latency
    INITIATION_INTERVALS = arch.initiationIntervals

    operands = sm.cores[threadID].operands[warp.registerPointer]
    operandLocks = sm.cores[threadID].operandLocks[warp.registerPointer]

    # Locks the wb, such that other insn that are dependant on this one will wait. 
    # 0 means there is no wb
    warp.wb != 0 && @yield request(operandLocks[warp.wb])



    # Assures all insn that could change the vals at the request addr are finished. 
    for addr in warp.operands
        if (addr isa Address)
            if (addr.value != warp.wb) # Data was already requested
                @yield request(operandLocks[addr.value])
                @yield unlock(operandLocks[addr.value])
            end
        end
    end

    # Seperate loop because all prior dependencies need to be resolved
    activeOperands = []
    for addr in warp.operands
        if (addr isa Address)
            push!(activeOperands, operands[addr.value])
        else
            push!(activeOperands, addr) # The address is the actual value
        end
    end

    ## LOADTID
    # Loads the TID in the wb adress.
    # valsOperands[1] == the blockIdx.x * blockDim.x value from the kernel/warp scheduler.

    if warp.type == "LOADTID"
        @yield timeout(sim, LATENCY[warp.type])
        operands[warp.wb] = threadID + activeOperands[1]
    elseif warp.type == "LOADC"
        @yield request(arch.SM.loadwriteBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type])
        @yield unlock(arch.SM.loadwriteBW)

        operands[warp.wb] = sm.constantMemory.data[activeOperands[1]]
    elseif warp.type == "LOADS"
        @yield request(arch.SM.loadwriteBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type])
        @yield unlock(arch.SM.loadwriteBW)

        operands[warp.wb] = arch.SM.sharedMemory.data[activeOperands[1]]
    elseif warp.type == "LOADG"
        requestedAddr = activeOperands[1]

        ## Cache logic
        # Check if the data is in L1 (is already loaded).
        # Also checks if it is already requested. Otherwise will issue a new request.
        #   L1 would be the vals in [globalMemoryReceived +- MEMREQUESTSIZE]

        if (minimum(abs.(requestedAddr .- arch.SM.globalMemoryReceived)) < MEMREQUESTSIZE)
            @yield timeout(sim, 2)
        else
            if (minimum(abs.(requestedAddr .- arch.SM.globalMemoryRequests)) < MEMREQUESTSIZE) # Because a load request loads also the 1024 next vals
                while minimum(abs.(requestedAddr .- arch.SM.globalMemoryReceived)) > MEMREQUESTSIZE
                    @yield timeout(sim, 1)
                end
            else
                push!(arch.SM.globalMemoryRequests, requestedAddr)
                @yield request(arch.SM.globalMemoryRequestBW)
                @yield timeout(sim, INITIATION_INTERVALS[warp.type])
                @yield unlock(arch.SM.globalMemoryRequestBW)
                @yield timeout(sim, LATENCY[warp.type] - INITIATION_INTERVALS[warp.type])
                push!(arch.SM.globalMemoryReceived, requestedAddr)
            end
        end
        operands[warp.wb] = arch.globalMemory.data[requestedAddr]

    elseif warp.type == "MULT"
        @yield request(arch.SM.intFUBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type]) # Pipelined so only 1 cycle timeout
        @yield unlock(arch.SM.intFUBW)
        @yield timeout(sim, LATENCY[warp.type] - INITIATION_INTERVALS[warp.type])

        operands[warp.wb] = activeOperands[1] * activeOperands[2]


    elseif warp.type == "ADD"
        @yield request(arch.SM.intFUBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type])
        @yield unlock(arch.SM.intFUBW)

        @yield timeout(sim, LATENCY[warp.type] - INITIATION_INTERVALS[warp.type])

        operands[warp.wb] = activeOperands[1] + activeOperands[2]

        # Will run the julia function supplied at runtime
    elseif length(warp.type) > 4 && warp.type[1:4] == "INT."
        @yield request(arch.SM.intFUBW)
        @yield timeout(sim, INITIATION_INTERVALS["ADD"])
        @yield unlock(arch.SM.intFUBW)

        @yield timeout(sim, LATENCY["ADD"] - INITIATION_INTERVALS["ADD"])

        command = warp.type[5:end]
        println("$(command)($(activeOperands)...)")
        out::Int = eval(Meta.parse("$(command)($(activeOperands)...)"))

        operands[warp.wb] = out
    elseif length(warp.type) > 5 && warp.type[1:5] == "JUMP."
        @yield request(arch.SM.intFUBW)
        @yield timeout(sim, INITIATION_INTERVALS["ADD"])
        @yield unlock(arch.SM.intFUBW)

        @yield timeout(sim, LATENCY["ADD"] - INITIATION_INTERVALS["ADD"])

        command = warp.type[5:end]
        commandOperands = activeOperands[1:2]
        testResult::Bool = eval(Meta.parse("$(command)($(commandOperands)...)"))

        if testResult
            verbosity > 4 && println("Masking $threadID with $(activeOperands[end])")
            sm.cores[threadID].mask += activeOperands[end]
        end

    elseif warp.type == "WBS"
        @yield request(arch.SM.loadwriteBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type])
        @yield unlock(arch.SM.loadwriteBW)

        sm.sharedMemory.data[activeOperands[1]] = activeOperands[2]
    else
        throw("Unknown command $(warp.type)")
    end

    warp.wb != 0 && @yield unlock(operandLocks[warp.wb])
    warp.threads -= 1
end

