using ResumableFunctions
using ConcurrentSim

include("pluto_exporter.jl")
println("=== Running tests ===")

## Creating dummy environment

const SAXPY_X = collect(1:1000) .+ 5
const SAXPY_Y = collect(2:2:2000)
const SAXPY_a = 5

function cleanEnv(registerSize=1)
    sim = Simulation()

    arch = Architecture(1, SM(sim), GlobalMemory(), [])

    arch.globalMemory.data[1:1000] = SAXPY_X # Vector X
    arch.globalMemory.data[1001:2000] = SAXPY_Y # Vector Y
    arch.SM.constantMemory.data[1] = SAXPY_a # Scalar a
    return sim, arch
end


## Test LOADTID
println("=== Test LOADTID ===")
@resumable function testTID(sim, arch)
    insn = Instruction("LOADTID", [], [0], 1)
    @process simulateWarp(sim, arch, arch.SM, insn)
end
sim, arch = cleanEnv();
@process testTID(sim, arch)
run(sim)
@assert arch.SM.cores[1].operands[1][1] == 1
@assert arch.SM.cores[32].operands[1][1] == 32


## Test LOADC
println("=== Test LOADC ===")
@resumable function testLOADC(sim, arch)
    insn = Instruction("LOADC", [], [1], 2)
    @process simulateWarp(sim, arch, arch.SM, insn)
end
sim, arch = cleanEnv();
@process testLOADC(sim, arch)
run(sim)
@assert arch.SM.cores[1].operands[1][2] == SAXPY_a # == 5
@assert arch.SM.cores[32].operands[1][2] == SAXPY_a # == 5

## Test LOADG relative to TID
println("=== Test LOADG ===")
@resumable function testLOADG(sim, arch)
    insn = Instruction("LOADG", [1], [], 3)
    @process simulateWarp(sim, arch, arch.SM, insn)
end
sim, arch = cleanEnv();
@process testTID(sim, arch)
@process testLOADG(sim, arch)
run(sim)
@assert arch.SM.cores[1].operands[1][3] == SAXPY_X[1] # == 1
@assert arch.SM.cores[32].operands[1][3] == SAXPY_X[32] # == 32
@assert 1 in arch.SM.globalMemoryReceived # Check wheter the request bundling works
@assert 2 ∉ arch.SM.globalMemoryReceived

## Test MULT relative to prevs
println("=== Test MULT ===")
@resumable function testMULT(sim, arch)
    insn = Instruction("MULT", [2, 3], [], 5)
    @process simulateWarp(sim, arch, arch.SM, insn)
end
sim, arch = cleanEnv();
@process testTID(sim, arch)
@process testLOADC(sim, arch)
@process testLOADG(sim, arch)
@process testMULT(sim, arch)

run(sim)
@assert arch.SM.cores[1].operands[1][5] == SAXPY_X[1] * 5
@assert arch.SM.cores[32].operands[1][5] == SAXPY_X[32] * 5


## Test ADD relative to prevs and load a global var from it
println("=== Test ADD ===")
@resumable function testADD(sim, arch)
    insn = Instruction("ADD", [1], [1000], 4) # Calculate address of y
    @process simulateWarp(sim, arch, arch.SM, insn)
    insn2 = Instruction("LOADG", [4], [], 6)
    @process simulateWarp(sim, arch, arch.SM, insn2)
end
sim, arch = cleanEnv();
@process testTID(sim, arch)
@process testADD(sim, arch)

run(sim)
@assert arch.SM.cores[1].operands[1][4] == arch.SM.cores[1].operands[1][1] + 1000
@assert arch.SM.cores[end].operands[1][4] == arch.SM.cores[end].operands[1][1] + 1000

@assert arch.SM.cores[1].operands[1][6] == SAXPY_Y[1] # == 2 (2:2:2000)
@assert arch.SM.cores[32].operands[1][6] == SAXPY_Y[32] # == 64


## Test WB to sharedMemory. It saves the TID at addres TID
println("=== Test WBS ===")
@resumable function testWBS(sim, arch)
    insn = Instruction("WBS", [1, 1], [], 1) # The wb actually doesn't get used
    @process simulateWarp(sim, arch, arch.SM, insn)
end
sim, arch = cleanEnv();
@process testTID(sim, arch)
@process testWBS(sim, arch)
run(sim)

@assert arch.SM.sharedMemory.data[1:32] == collect(1:32)
@assert arch.SM.sharedMemory.data[33] == 0

## Test SAXPY insn's completely for 1 loop

println("=== Test SAXPY ===")
@resumable function testSAXPY(sim, arch)
    @process testTID(sim, arch)
    @process testLOADC(sim, arch)
    @process testLOADG(sim, arch)

    @process testMULT(sim, arch)

    @process testADD(sim, arch)

    insn = Instruction("ADD", [5, 6], [], 7)
    @process simulateWarp(sim, arch, arch.SM, insn)

    insn2 = Instruction("WBS", [Address(1), Address(7)], 0) # Write the solution to shared memory at index TID
    @process simulateWarp(sim, arch, arch.SM, insn2)

end
sim, arch = cleanEnv();

@process testSAXPY(sim, arch)

run(sim, 1000)

if (false)
    println(["tid", "a", "x", "yaddr", "a*x", "y", "result"])
    for i in 1:32
        print(arch.SM.cores[i].operands[1][1:7])
        print(" : ")
        println(SAXPY_a * SAXPY_X[i] + SAXPY_Y[i])
    end
end

for i in 1:32
    @assert arch.SM.cores[i].operands[1][7] == SAXPY_a * SAXPY_X[i] + SAXPY_Y[i]
    @assert arch.SM.sharedMemory.data[i] == SAXPY_a * SAXPY_X[i] + SAXPY_Y[i] "Problem with WB"
end

## Test custom INT instructions (div() could be any Julia function)
println("=== Test INT.div insn ===")
@resumable function testINT(sim, arch)
    insn = Instruction("INT.div", [Address(3), Address(2)], 5)
    insn2 = Instruction("INT.factorial", [5], 6)
    @process simulateWarp(sim, arch, arch.SM, insn)
    @process simulateWarp(sim, arch, arch.SM, insn2)
end
sim, arch = cleanEnv();
@process testTID(sim, arch)
@process testLOADC(sim, arch)
@process testLOADG(sim, arch)
@process testINT(sim, arch)

run(sim)
@assert arch.SM.cores[1].operands[1][5] == div(SAXPY_X[1], 5)
@assert arch.SM.cores[32].operands[1][5] == div(SAXPY_X[32], 5)
@assert arch.SM.cores[5].operands[1][6] == factorial(5)

## Test custom JUMP instructions
println("=== Test JUMP insn ===")
function loadJUMP(arch)
    insns = [
        Instruction("LOADTID", [], [0], 1),
        Instruction("JUMP.>=", [Address(1), 10, 2]), # Jump 2 instructions if TID > 10
        Instruction("JUMP.==", [Address(1), 3, 1]), # Jump 1 instruction if TID == 15
        Instruction("INT.^", [2, Address(1)], 5)
    ]
    push!(arch.kernel, insns...)
end

sim = Simulation()
arch = Architecture(1, SM(sim, 32, 1), GlobalMemory(), [], 16);

loadJUMP(arch)

@process simulate(sim, arch, verbosity=3)
run(sim, 50000)

@assert [arch.SM.cores[i].operands[1][5] for i in 1:4] == [2, 4, 0, 16]
@assert arch.SM.cores[32].operands[1][5] == 0

println([arch.SM.cores[i].operands[1][5] for i in 1:32])

printstyled("\nAll tests finished succesfully\n", color=:green)