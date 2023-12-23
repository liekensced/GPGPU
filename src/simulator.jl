using ConcurrentSim
using ResumableFunctions

# include("structs.jl")


@resumable function simulate(sim::Simulation, arch::Architecture)
    
    

    
    # First the warp scheduler acts every cycle:
    scheduledWarps = 0;
    for warp in arch.kernel
        @yield request(arch.SM.insnBW)
        @yield timeout(sim, 1) # Timeout for the warp to be scheduled.
        if (scheduledWarps == 8) # The 8 could be enhanced to blockSize. Assures lockstep.
            while length(arch.SM.warpBuffer) > 0
                @yield timeout(sim, 1)
            end
            scheduledWarps = 0;
        end
        scheduledWarps += 1;

        @yield unlock(arch.SM.insnBW)
        @process simulateWarp(sim, arch, arch.SM, warp) # Yielding insnBW implies a free warp in this case.
        # intentionally not yielded
    end
end

@resumable function simulateWarp(sim::Simulation, arch, sm::SM, warp::Instruction)
    push!(sm.warpBuffer, warp)
    startTime = now(sim);
    for threadID in 1:sm.coreAmount
        @process simulateThread(sim, arch, sm, warp, threadID)
    end

    while warp.threads > 0
        @yield timeout(sim, 1)
    end

    print("Executed warp: \e[32m $(warp.type)\e[0m @ $(now(sim)) \e[90m")
    println([insn.type for insn in arch.SM.warpBuffer])
    print("\e[0m")

    push!(arch.hist, InstructionRecord(startTime,now(sim),warp))
    deleteat!(sm.warpBuffer, findfirst(x -> x == warp, sm.warpBuffer))
end

@resumable function simulateThread(sim::Simulation, arch, sm::SM, warp::Instruction, threadID)
    
    MEMREQUESTSIZE = arch.memRequestSize;
    LATENCY = arch.latency
    INITIATION_INTERVALS = arch.initiationIntervals
    
    operands = sm.cores[threadID].operands


    @yield request(sm.cores[threadID].operandLocks[warp.wb])


    for addr in warp.addrOperands

        addr == warp.wb && continue

        @yield request(sm.cores[threadID].operandLocks[addr])
        @yield unlock(sm.cores[threadID].operandLocks[addr])
    end



    ## LOADTID
    # Loads the TID in the wb adress.
    # valsOperands[1] == the blockIdx.x * blockDim.x value from the kernel/warp scheduler.

    if warp.type == "LOADTID"
        @yield timeout(sim, LATENCY[warp.type])
        operands[warp.wb] = threadID + warp.valsOperands[1]
    elseif warp.type == "LOADC"
        @yield request(arch.SM.loadwriteBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type]) # Timeout for the warp to be scheduled.
        @yield unlock(arch.SM.loadwriteBW)

        operands[warp.wb] = sm.constantMemory.data[warp.valsOperands[1]]
    elseif warp.type == "LOADG"
        requestedAddr = operands[warp.addrOperands[1]]
        # Check if it is already loaded
        
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
                @yield timeout(sim, LATENCY[warp.type]-INITIATION_INTERVALS[warp.type])
                push!(arch.SM.globalMemoryReceived, requestedAddr)
            end
        end
        operands[warp.wb] = arch.globalMemory.data[requestedAddr]

    elseif warp.type == "MULT"
        @yield request(arch.SM.intFUBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type]) # Pipelined so only 1 cycle timeout
        @yield unlock(arch.SM.intFUBW)
        @yield timeout(sim, LATENCY[warp.type]-INITIATION_INTERVALS[warp.type])

        if (length(warp.addrOperands) == 2)
            operands[warp.wb] = operands[warp.addrOperands[1]] * operands[warp.addrOperands[2]]
        elseif (length(warp.addrOperands) == 1)
            operands[warp.wb] = operands[warp.addrOperands[1]] * warp.valsOperands[1]
        elseif (length(warp.valsOperands) == 2)
            operands[warp.wb] = warp.valsOperands[1] * warp.valsOperands[2]
        else
            throw("mult error")
        end


    elseif warp.type == "ADD"
        @yield request(arch.SM.intFUBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type])
        @yield unlock(arch.SM.intFUBW)

        @yield timeout(sim, LATENCY[warp.type]-INITIATION_INTERVALS[warp.type])

        if (length(warp.addrOperands) == 2)
            operands[warp.wb] = operands[warp.addrOperands[1]] + operands[warp.addrOperands[2]]
        elseif (length(warp.addrOperands) == 1)
            operands[warp.wb] = operands[warp.addrOperands[1]] + warp.valsOperands[1]
        elseif (length(warp.valsOperands) == 2)
            operands[warp.wb] = warp.valsOperands[1] + warp.valsOperands[2]
        else
            throw("ADD error")
        end

    elseif warp.type == "WBS"
        @yield request(arch.SM.loadwriteBW)
        @yield timeout(sim, INITIATION_INTERVALS[warp.type])
        @yield unlock(arch.SM.loadwriteBW)

        sm.sharedMemory.data[operands[warp.addrOperands[1]]] = operands[warp.addrOperands[2]]

    end

    @yield unlock(sm.cores[threadID].operandLocks[warp.wb])
    warp.threads -= 1
end