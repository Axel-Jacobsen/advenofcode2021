#! /usr/local/bin/julia


function quantify(data; predicate = x -> x && true)
    mapreduce(predicate, +, data)
end


function p1()
    open("inputs/d01.txt", "r") do io
        data = map(s -> parse(Int64, s), eachline(io))
        return quantify(zip(data, data[2:end]), predicate = ((i, j),) -> j > i)
    end
end


function p2()
    open("inputs/d01.txt", "r") do io
        data = map(s -> parse(Int64, s), eachline(io))
        return quantify(
            zip(
                zip(data, data[2:end], data[3:end]),
                zip(data[2:end], data[3:end], data[4:end]),
            ),
            predicate = ((i, j),) -> sum(j) > sum(i),
        )
    end
end

println(p1())
println(p2())
