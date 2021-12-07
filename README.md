# Advent of Code, 2021!

It is truly the most wonderful time of the year! Here is my work for this year's Advent.

I am doing it in Julia, Python, Rust (slowly), and hopefully another, more functional language for some problems.

Folders hold solutions in languages other than Julia or Python. This is the Julia notebook. Enjoy!


```julia
using JupyterFormatter;
enable_autoformat();

# Helpers, of course
function quantify(data; predicate = x -> x)
    mapreduce(predicate, +, data)
end


function process_inputs(day::String; convert::Function = s -> parse(Int64, s))
    open("inputs/d$day.txt", "r") do io
        map(s -> convert(s), eachline(io))
    end
end
```




    process_inputs (generic function with 1 method)



## Day 1!


```julia
function p1()
    data = process_inputs("01")
    quantify(zip(data, data[2:end]), predicate = ((i, j),) -> j > i)
end


function p2()
    data = process_inputs("01")
    quantify(zip(data, data[4:end]), predicate = ((i, j),) -> sum(j) > sum(i))
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

data = process_inputs("02", convert = conv_inp)


function p1()
    x = z = 0
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
    x * z
end


function p2()
    x = z = aim = 0
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
    x * z
end

println(p1())
println(p2())
```

    1499229
    1340836560


## Day 3!


```julia
data = process_inputs("03", convert = x -> map(s -> parse(Int64, s), split(x, "")))

function to_int(arr::Array{T})::Int64 where {T<:Number}
    s = 0
    for i = length(arr):-1:1
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
    for i = 1:length(data[1])
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



## Day 4!


```julia
mutable struct Board{T<:Integer}
    board::Matrix{T}
    called_num_idxs::Matrix{T}
    Board(board_matrix::Matrix{T}) where {T<:Real} =
        new{T}(board_matrix, ones(T, size(board_matrix)))
end


Base.show(io::IO, b::Board) =
    print(io, "Board(board=$(b.board), called_nums=$(b.called_num_idxs))")


function call_number(b::Board, num::T)::Bool where {T<:Integer}
    idx = findfirst(n -> n == num, b.board)
    if idx == nothing
        return false
    end
    b.called_num_idxs[idx] = 0
    return true
end


function board_has_win(b::Board)::Bool
    for row in eachrow(b.called_num_idxs)
        if row == zeros(5)
            return true
        end
    end

    for col in eachcol(b.called_num_idxs)
        if col == zeros(5)
            return true
        end
    end

    return false
end


function get_day4_inputs()::Tuple{Array{Int64},Array{Board}}
    open("inputs/d04.txt", "r") do io
        bingo_nums = map(s -> parse(Int64, s), split(readline(io), ","))
        rest = strip(read(io, String))

        boards = []
        for board in split(rest, "\n\n")
            board_matrix = zeros(Int64, 5, 5)
            for (i, row) in enumerate(split(board, "\n"))
                # there is probably a cleaner regex, but this is A-OK w/ me
                row_regex = r"\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)"
                row_captures = match(row_regex, row).captures
                row_integers = map(s -> parse(Int64, s), row_captures)
                board_matrix[i, :] .= row_integers
            end
            push!(boards, Board(board_matrix))
        end
        return (bingo_nums, boards)
    end
end


function p1()
    bingo_nums, boards = get_day4_inputs()
    for num in bingo_nums
        map(b -> call_number(b, num), boards)
        maybe_winning_board_idx = findfirst(board_has_win, boards)
        if maybe_winning_board_idx != nothing
            winning_board = boards[maybe_winning_board_idx]
            return num * sum(winning_board.board .* winning_board.called_num_idxs)
        end
    end
    throw("no board won")
end


function p2()
    bingo_nums, boards = get_day4_inputs()
    num_winning_boards = 0
    winning_boards = Set{Board}()

    for num in bingo_nums
        foreach(b -> call_number(b, num), boards)

        recent_winning_board_idxs =
            findall(b -> board_has_win(b) && !in(b, winning_boards), boards)

        foreach(b -> push!(winning_boards, b), boards[recent_winning_board_idxs])

        if num_winning_boards == length(boards) - 1 &&
           length(recent_winning_board_idxs) == 1
            winning_board = boards[pop!(recent_winning_board_idxs)]
            return num * sum(winning_board.board .* winning_board.called_num_idxs)
        else
            num_winning_boards += length(recent_winning_board_idxs)
        end
    end
end


p1(), p2()
```




    (67716, 1830)


