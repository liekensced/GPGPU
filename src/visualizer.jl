try
    using NativeSVG
catch
    Pkg.add(url="https://github.com/BenLauwens/NativeSVG.jl.git")
end


function visualize(arch::Architecture, showStats=false)
    SCALEFACTOR = 2
    colors = Dict([("LOADTID", "black"), ("LOADC", "orange"), ("LOADG", "red"), ("MULT", "darkBlue"), ("ADD", "blue"), ("WBS", "purple")])

    LATENCY = arch.latency
    INITIATION_INTERVALS = arch.initiationIntervals

    sort!(arch.hist, by=x -> x.startTime)
    i = 0

    if (showStats)
        _,amount,avg = stats(arch)
    else
        amount,avg = [],[]
    end

    maxHeight = (length(arch.hist)+1+length(amount))*10
    maxWidth = (arch.hist[end].endTime+1)*SCALEFACTOR


    Drawing(height = maxHeight,width = maxWidth) do
        for hist in arch.hist
            i += 1
            line(x1=hist.startTime * SCALEFACTOR, y1=i * 10, x2=hist.endTime * SCALEFACTOR, y2=i * 10, stroke=colors[hist.insn.type], stroke_width="10")
            if (hist.insn.type == "LOADG")
                line(x1=string((hist.endTime - INITIATION_INTERVALS[hist.insn.type]) * SCALEFACTOR), y1=i * 10, x2=string(hist.endTime * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="3")
                
                if hist.endTime - hist.startTime > 50
                    line(x1=string((hist.endTime - 50) * SCALEFACTOR), y1=i * 10, x2=string(hist.endTime * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="1")
                end
            else
                line(x1=string((hist.endTime - LATENCY[hist.insn.type]) * SCALEFACTOR), y1=i * 10, x2=string(hist.endTime * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="3")
            end
        end
        for type in keys(avg)
            i += 1
            line(x1=0, y1=i*10, x2= avg[type] * SCALEFACTOR, y2=i * 10, stroke=colors[type], stroke_width="10")
            line(x1=0, y1=i * 10, x2=string(LATENCY[type] * SCALEFACTOR), y2=i * 10, stroke="black", stroke_width="3")
        end
    end

end


function stats(arch::Architecture)
    amount = Dict([("LOADTID", 0), ("LOADC", 0), ("LOADG", 0), ("MULT", 0), ("ADD", 0), ("WBS", 0)])
    avg = Dict{String, Float64}([("LOADTID", 0), ("LOADC", 0), ("LOADG", 0), ("MULT", 0), ("ADD", 0), ("WBS", 0)])


    for record in arch.hist
        insn = record.insn
        amount[insn.type] += 1
        avg[insn.type] += record.endTime - record.startTime
    end

    out = "";

    for type in keys(avg)
        avg[type] /= amount[type]
        out = out*type*": "*string(avg[type])*"\n"
    end

    out,amount,avg
end
