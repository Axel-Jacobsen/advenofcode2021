# Advent of Code, 2021!

It is truly the most wonderful time of the year! Here is my work for this year's Advent.

I am doing it in Julia, Python, Rust (slowly), and hopefully another, more functional language for some problems.

Folders hold solutions in languages other than Julia or Python. This is the Julia notebook. Enjoy!


```julia
# Helpers, of course

function quantify(data; predicate = x -> x)
    mapreduce(predicate, +, data)
end


function process_inputs(day::String; convert::Function=s -> parse(Int64, s))
    open("inputs/d$day.txt", "r") do io
        map(s -> convert(s), eachline(io))
    end
end
```




    process_inputs (generic function with 1 method)



## Day 1!


```julia
function p1()
    data = process_inputs("01");
    quantify(zip(data, data[2:end]), predicate = ((i, j),) -> j > i)
end


function p2()
    data = process_inputs("01");
    quantify(
            zip(data, data[4:end]),
            predicate = ((i, j),) -> sum(j) > sum(i),
        )
end

println(p1())
println(p2())
```

    1316
    1344


## Day 2!


```julia
function conv_inp(inp)
    s = split(inp, " ")
    (s[1], parse(Int64, (s[2])))
end

data = process_inputs("02", convert=conv_inp)


function p1()
    x = z = 0;
    for (dir, val) in data
        if dir == "forward"
            x += val
        elseif dir == "back"
            x -= val
        elseif dir == "up"
            z -= val
        elseif dir == "down"
            z += val
        end
    end
    x*z
end


function p2()
    x = z = aim = 0;
    for (dir, val) in data
        if dir == "forward"
            x += val
            z += aim * val
        elseif dir == "up"
            aim -= val
        elseif dir == "down"
            aim += val
        end
    end
    x*z
end

println(p1())
println(p2())
```

    1499229
    1340836560


## Day 3!


```julia
data = process_inputs("03", convert = x -> map(
        s -> parse(Int64, s),
        split(x, "")
    ))

function to_int(arr::Array{T})::Int64 where {T<:Number}
    s = 0
    for i in length(arr):-1:1
        s += arr[i] * 2^(length(arr) - i)
    end
    s
end

function p1(data)
    ratios = sum(data) / length(data)
    gamma = map(x -> x >= 0.5, ratios)
    delta = map(x -> !x, gamma)
    to_int(gamma) * to_int(delta)
end

function p2(data)
    gammas = copy(data)
    deltas = copy(data)
    for i in 1:length(data[1])
        gamma_rate = sum(gammas) / length(gammas)
        delta_rate = sum(deltas) / length(deltas)

        if length(gammas) > 1
            filter!(num -> num[i] == (gamma_rate[i] >= 0.5), gammas)
        end
        if length(deltas) > 1
            filter!(num -> num[i] == (delta_rate[i] < 0.5), deltas)
        end
    end
    to_int(gammas[1]) * to_int(deltas[1])
end

p1(data), p2(data)
```




    (1307354, 482500)


