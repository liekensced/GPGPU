using ConcurrentSim

const ConstantMemorySIZE = 1000;
const SharedMemorySIZE = 16000;
const GlobalMemorySIZE = 64000;

const DEFAULT_MEMREQUESTSIZE = 64;
const DEFAULT_LATENCY = Dict([("LOADTID", 1),("LOADC",4),("LOADG",50),("MULT",5),("ADD",3),("WBS",2)])
const DEFAULT_INITIATION_INTERVALS = Dict([("LOADTID", 0),("LOADC",4),("LOADG",2),("MULT",1),("ADD",1),("WBS",2)])


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

mutable struct CUDACore
    operands::Vector{Int}
    operandLocks::Vector{Resource}
    CUDACore(sim::Simulation, size=20) = new(zeros(Number, size), [Resource(sim, 1) for _ in 1:size])
end

mutable struct Instruction

    type::String
    addrOperands::Vector{Int} # Adresses are just integers, relative to it's TID
    valsOperands::Vector{Number}

    wb::Int # Write Back adress

    threads::Int
    Instruction(type, addrOperands, valsOperands, wb) = new(type, addrOperands, valsOperands, wb, 32)
end

mutable struct SM
    coreAmount::Int # Amount of CUDA_Cores
    cores::Vector{CUDACore}
    sharedMemory::SharedMemory # Or register file, a few kilobytes, 
    constantMemory::ConstantMemory

    globalMemoryRequests::Vector{Int} 
    globalMemoryReceived::Vector{Int} # SharedMemory or L1 Cache

    #threads::Vector{Instruction} # The warp scheduler will bundle these threads in groups of 32 threads in SIMD fashion, we will hard code this into warps
    warpBuffer::Vector{Instruction}
    available::Int

    insnBW::Resource
    loadwriteBW::Resource
    intFUBW::Resource
    FPFUBW::Resource
    globalMemoryRequestBW::Resource
    SM(sim::Simulation, coreAmount=32) = new(coreAmount, [CUDACore(sim) for _ in 1:coreAmount], SharedMemory(), ConstantMemory(), [GlobalMemorySIZE], [GlobalMemorySIZE], Instruction[], 0, Resource(sim, 1), Resource(sim, coreAmount), Resource(sim, coreAmount), Resource(sim, coreAmount), Resource(sim, 1))
end


struct InstructionRecord
    startTime::Int
    endTime::Int
    insn::Instruction
end

mutable struct Architecture
    SMamount::Int # Amount of SM's
    SM::SM # Would be a vector of SM's
    globalMemory::GlobalMemory
    kernel::Vector{Instruction}

    memRequestSize::Int64
    latency::Dict{String,Int}
    initiationIntervals::Dict{String,Int}


    hist::Vector{InstructionRecord}

    Architecture(SMamount, SM, globalMemory, kernel, memRequestSize=DEFAULT_MEMREQUESTSIZE,latency=DEFAULT_LATENCY,initiationIntervals=DEFAULT_INITIATION_INTERVALS, hist=InstructionRecord[]) = new(SMamount, SM, globalMemory, kernel,memRequestSize,latency,initiationIntervals, hist)
end
