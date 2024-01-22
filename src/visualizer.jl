try
    using NativeSVG
    using PrettyTables
catch
    Pkg.add(url="https://github.com/BenLauwens/NativeSVG.jl.git")
    Pkg.add("PrettyTables")
end


function visualize(arch::Architecture, showStats::Bool=false, stopWarp::Int64=99999)
    SCALEFACTOR = 2
    colors = Dict([("LOADTID", "black"), ("LOADC", "orange"), ("LOADG", "red"), ("LOADS", "red"), ("MULT", "darkBlue"), ("ADD", "blue"), ("WBS", "purple")])
    resourceColors = ["white", "orange", "blue", "red"]

    LATENCY = arch.latency
    INITIATION_INTERVALS = arch.initiationIntervals
    sort!(arch.hist, by=x -> x.endTime)

    currentHists = arch.hist[1:minimum([length(arch.hist), stopWarp])]
    sort!(currentHists, by=x -> x.startTime)
    i = 0

    if (showStats)
        _, amount, avg = stats(arch)
    else
        amount, avg = [], []
    end

    maxHeight = (length(currentHists) + 1 + length(amount)) * 10
    maxWidth = maximum(map(x -> x.endTime + 1, currentHists)) * SCALEFACTOR

    if (!isempty(arch.usage))
        maxHeight += length(arch.usage[1]) * 10
    end


    Drawing(height=maxHeight, width=maxWidth) do
        for hist in currentHists
            i += 1
            insnType = hist.insn.type
            if length(insnType) > 4 && insnType[1:4] == "INT."
                insnType = "MULT"
            elseif length(insnType) > 5 && insnType[1:5] == "JUMP."
                insnType = "MULT"
            end


            line(x1=hist.startTime * SCALEFACTOR, y1=i * 10, x2=hist.endTime * SCALEFACTOR, y2=i * 10, stroke=colors[insnType], stroke_width="10")
            if (hist.insn.type == "LOADG")
                line(x1=string((hist.endTime - INITIATION_INTERVALS[insnType]) * SCALEFACTOR), y1=i * 10, x2=string(hist.endTime * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="3")

                if hist.endTime - hist.startTime > 50
                    line(x1=string((hist.endTime - 50) * SCALEFACTOR), y1=i * 10, x2=string(hist.endTime * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="1")
                end
            else
                line(x1=string((hist.endTime - LATENCY[insnType]) * SCALEFACTOR), y1=i * 10, x2=string(hist.endTime * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="3")
            end
        end
        if (!isempty(arch.usage))
            j = 0
            line(x1=0, y1=(i + 0.5 + length(arch.usage[1]) / 2) * 10, x2=length(arch.usage) * SCALEFACTOR, y2=(i + 0.5 + length(arch.usage[1]) / 2) * 10, stroke="black", stroke_width=(length(arch.usage[1])) * 10)
            for currentUsage in arch.usage
                j += 1
                iResource = 0
                for resourceUsage in currentUsage
                    iResource += 1
                    line(x1=j * SCALEFACTOR, y1=(i + iResource) * 10, x2=(j + 1) * SCALEFACTOR, y2=(i + iResource) * 10, stroke=resourceColors[iResource], stroke_width=resourceUsage * 10)
                end
            end
            i += length(arch.usage[1])
        end
        for type in keys(avg)
            i += 1
            line(x1=0, y1=i * 10, x2=avg[type] * SCALEFACTOR, y2=i * 10, stroke=colors[type], stroke_width="10")
            line(x1=0, y1=i * 10, x2=string(LATENCY[type] * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="3")
        end
    end

end


"""
    stats(arch::Architecture)

Not that interesting
"""
# function stats(arch::Architecture)
#     amount = Dict([("LOADTID", 0), ("LOADC", 0), ("LOADG", 0), ("LOADS", 0), ("MULT", 0), ("ADD", 0), ("WBS", 0)])
#     avg = Dict{String,Float64}([("LOADTID", 0), ("LOADC", 0), ("LOADG", 0), ("LOADS", 0), ("MULT", 0), ("ADD", 0), ("WBS", 0)])

#     insnType = hist.insn.type
#     if length(insnType) > 4 && insnType[1:4] == "INT."
#         insnType = "MULT"
#     elseif length(insnType) > 5 && insnType[1:5] == "JUMP."
#         insnType = "MULT"
#     end

#     for record in arch.hist
#         insn = record.insn
#         amount[insnType] += 1
#         avg[insnType] += record.endTime - record.startTime
#     end

#     out = ""

#     for type in keys(avg)
#         avg[type] /= amount[type]
#         out = out * type * ": " * string(avg[type]) * "\n"
#     end

#     out, amount, avg
# end

function visualizeData(arch::Architecture, i::Int; headers=[], headersTitles=[], sharedMemoryLines=1, width=32)
    record::InstructionRecord = arch.hist[minimum([i, length(arch.hist)])]
    headers = [x[1:width] for x in headers]
    operandsData = [x[1:width] for x in record.operands]
    sharedMemData = [record.sharedMemory[(1+width*(i-1)):(width*i)] for i in 1:sharedMemoryLines]

    table = []
    push!(table, headers...)
    push!(table, operandsData...)
    push!(table, sharedMemData...)

    labels = [headersTitles..., 1:length(operandsData)..., ["sharedMem" .* string(i) for i in 1:sharedMemoryLines]...]

    res = reshape([(table...)...], 32, length(labels))
    pretty_table(HTML, res', header=collect(1:width), row_labels=labels)
end
