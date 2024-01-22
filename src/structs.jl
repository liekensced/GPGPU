using ConcurrentSim

const ConstantMemorySIZE = 1000; # Will just throw an error if program exceeds these limits
const SharedMemorySIZE = 16000;
const GlobalMemorySIZE = 64000;

const DEFAULT_MEMREQUESTSIZE = 64; # Simulates the L1 cache of the global data
const DEFAULT_LATENCY = Dict([("LOADTID", 1),("LOADC",4),("LOADG",50),("MULT",5),("ADD",3),("WBS",2),("LOADS",0)])
const DEFAULT_INITIATION_INTERVALS = Dict([("LOADTID", 0),("LOADC",4),("LOADG",2),("MULT",1),("ADD",1),("WBS",2),("LOADS",1)])
# Inspired from https://gpgpu-sim.org


"""
Integer wrapper to indicate it should be interpreted as an address
"""
struct Address
    value::Int64
    function Address(val::Int64)
         new(val)
    end
end

"""
Memory shared between the CUDA Cores on an SM
Modeled as a Vector where the address is the index
"""
mutable struct SharedMemory
    data::Vector
    SharedMemory() = new(zeros(Number, SharedMemorySIZE))
end

mutable struct ConstantMemory
    # Read Only
    data::Vector
    ConstantMemory() = new(zeros(Number, ConstantMemorySIZE))
end

mutable struct GlobalMemory
    data::Vector
    GlobalMemory() = new(zeros(Number, GlobalMemorySIZE))
end

"""
    Contains the state of a Core.
        note `SM.intFUBW::Resource` limits the amount of cores used
"""
mutable struct CUDACore
    operands::Vector{Vector{Int}} # vector of operand contexts.
    operandLocks::Vector{Vector{Resource}}

    mask::Int64
    CUDACore(sim::Simulation, registerSize=2, size=10) = new([zeros(Number, size) for _ in 1:registerSize], [[Resource(sim, 1) for _ in 1:size] for _ in 1:registerSize], 0)
end

"""
The instruction the warp should execute.
Instruction(type, operands, writeback, registerPointer)

    "LOADTID" [blockIdx.x * blockDim.x]
    "LOADC" [constant memory addres]
    "LOADS" [shared memory address]
    "LOADG" [global memory address]
    "ADD" [1, 2] => [1]+[2]
    "MULT" [1, 2] => [1]*[2]
    "WBS" [shared memory address, value]
    "JUMP.<exp>" [1, 2, Amount of insn to jump] => <expr>([1],[2])
    "INT.<exp>" [args...] => <expr>(args...)
"""
mutable struct Instruction

    type::String
    operands::Vector{Union{Int, Address}} # Adresses are wrapped in an Address struct, relative to it's TID

    wb::Int # Write Back adress
    
    registerPointer::Int # Determines which operand context to use

    threads::Int # Starts at the amount of cores. Will diminish when a thread is finished.

    # Deprecated
    function Instruction(type, addrOperands::Vector, valsOperands::Vector, wb, registerPointer=1)
        operands::Vector{Union{Int, Address}} = valsOperands
        for val in addrOperands
            push!(operands, Address(val))
        end
        new(type, operands, wb,registerPointer, 32)
    end
    Instruction(type, operands::Vector, wb=0, registerPointer=1) = new(type, operands, wb,registerPointer, 32)
end

"""
    Streaming Multiprocessor. Will execute warps in warpBuffer.
    SM(
        sim,
        amount of cores,
        register size = how many operand contexts it can hold
    )
"""
mutable struct SM
    coreAmount::Int # Amount of CUDA_Cores
    cores::Vector{CUDACore}
    sharedMemory::SharedMemory # a few kilobytes, 
    constantMemory::ConstantMemory

    globalMemoryRequests::Vector{Int} 
    globalMemoryReceived::Vector{Int} # Simulates L1 Cache

    #threads::Vector{Instruction} # The warp scheduler will bundle these threads in groups of 32 threads in SIMD fashion, we will hard code this into warps
    warpBuffer::Vector{Instruction}
    available::Int

    insnBW::Resource # Limits the amount of Instructions that can be issued at a given time
    loadwriteBW::Resource 
    intFUBW::Resource # Integer Functional Unit BandWidth
    FPFUBW::Resource# Floating Point Functional Unit BandWidth (currently not used)
    globalMemoryRequestBW::Resource
    SM(sim::Simulation, coreAmount=32, registerSize=1) = new(coreAmount, [CUDACore(sim,registerSize) for _ in 1:coreAmount], SharedMemory(), ConstantMemory(), [GlobalMemorySIZE], [GlobalMemorySIZE], Instruction[], 0, Resource(sim, 1), Resource(sim, coreAmount), Resource(sim, coreAmount), Resource(sim, coreAmount), Resource(sim, 1))
end

"""
    For logging purposes. Saves a copy of the state at endTime. 
        `operands` is a vector of first what operand and secondly the value in each thread
        * Note it is saved differently on the cores.
        * Note it could be empty if it isn't logged

    See Architecture.usage for a record of Resource usage
"""
struct InstructionRecord
    startTime::Int
    endTime::Int
    insn::Instruction
    operands::Vector{Vector{Int}} # Which operand -> 32 cores
    sharedMemory::Vector{Int}
end

"""
    Architecture(
        the amount of SM's,
        SM structs,
        initial globalMemory,
        kernel,

        memRequestSize, latency, initiationIntervals
        )
"""
mutable struct Architecture
    SMamount::Int # Amount of SM's
    SM::SM # Would be a vector of SM's
    globalMemory::GlobalMemory
    kernel::Vector{Instruction}

    memRequestSize::Int64 # How much memory is stored in the cache
    latency::Dict{String,Int}
    initiationIntervals::Dict{String,Int}


    hist::Vector{InstructionRecord}

    usage::Vector{Vector{}} # Used to save a record of the Resources level at every time. Could be empty!

    running::Bool # False when all warps are finished.

    Architecture(SMamount, SM, globalMemory, kernel, memRequestSize=DEFAULT_MEMREQUESTSIZE,latency=DEFAULT_LATENCY,initiationIntervals=DEFAULT_INITIATION_INTERVALS, hist=InstructionRecord[]) = new(SMamount, SM, globalMemory, kernel,memRequestSize,latency,initiationIntervals, hist,[],true)
end
