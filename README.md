# Advent of Code, 2021!

It is truly the most wonderful time of the year! Here is my work for this year's Advent.

I am doing it in Julia, Python, Rust (slowly), and hopefully another, more functional language for some problems.

Folders hold solutions in languages other than Julia or Python. This is the Julia notebook. Enjoy!


```julia
using JupyterFormatter;
enable_autoformat();

# Helpers, of course
function quantify(predicate::Function, data)
    mapreduce(predicate, +, data)
end


function process_inputs(day::String)
    open("inputs/d$day.txt", "r") do io
        map(s -> parse(Int64, s), eachline(io))
    end
end

function process_inputs(convert::Function, day::String)
    open("inputs/d$day.txt", "r") do io
        map(s -> convert(s), eachline(io))
    end
end

mutable struct ComboIterator
    d::Vector
    i::Int
    j::Int
    ComboIterator(d) = length(d) <= 2 ? nothing : new(d, 1, 2)
end

Base.eltype(::Type{ComboIterator}) = Tuple

function Base.length(itr::ComboIterator)::Int64
    n = length(itr.d)
    bottom_triangle = (n - itr.i - 1) * (n - itr.i) / 2
    current_row = n - itr.j
    bottom_triangle + current_row + 1
end

function Base.iterate(itr::ComboIterator, state = (1, 2))
    N = length(itr.d)
    (i, j) = state

    next_i = i
    next_j = j + 1
    if i == N
        return nothing
    elseif j == N
        next_i = i + 1
        next_j = i + 2
    end
    (itr.i, itr.j) = (next_i, next_j)
    (itr.d[i], itr.d[j]), (next_i, next_j)
end
```

## Day 1!


```julia
function p1()
    data = process_inputs("01")
    quantify(((i, j),) -> j > i, zip(data, data[2:end]))
end


function p2()
    data = process_inputs("01")
    quantify(((i, j),) -> j > i, zip(data, data[4:end]))
end

p1(), p2()
```




    (1316, 1344)



## Day 2!


```julia
function conv_inp(inp)
    s = split(inp, " ")
    (s[1], parse(Int64, (s[2])))
end

data = process_inputs(conv_inp, "02")


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
function to_int(arr::Array{T})::Int64 where {T<:Number}
    s = 0
    for i = length(arr):-1:1
        s += arr[i] * 2^(length(arr) - i)
    end
    s
end

function p1()
    data = process_inputs(x -> map(s -> parse(Int64, s), split(x, "")), "03")
    ratios = sum(data) / length(data)
    gamma = map(x -> x >= 0.5, ratios)
    delta = map(x -> !x, gamma)
    to_int(gamma) * to_int(delta)
end

function p2()
    data = process_inputs(x -> map(s -> parse(Int64, s), split(x, "")), "03")
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

p1(), p2()
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



## Day 5!

I gotta think harder about this one. It is probably simple but for some reason its been giving me trouble, and hasn't been as fun as the others. I'll come back to it once I have a break in finals.


```julia
is_integer(x::T) where {T<:Number} = floor(Int, x) == x

struct Point{T<:Integer}
    x::T
    y::T
end

struct LineSegment{T<:Integer}
    P1::Point{T}
    P2::Point{T}
end

function get_d5_data()::Vector{LineSegment}
    data = process_inputs("05test") do s
        r1_str, r2_str = split(s, " -> ")
        x1, y1 = map(v -> parse(Int64, v), split(r1_str, ","))
        x2, y2 = map(v -> parse(Int64, v), split(r2_str, ","))
        x1 ≤ x2 ? LineSegment(Point(x1, y1), Point(x2, y2)) :
        LineSegment(Point(x2, y2), Point(x1, y2))
    end
    sort(data, by = LS -> LS.P1.x)
end

function segments_intersect(L1::LineSegment, L2::LineSegment)::Bool
    tn =
        (L1.P1.x - L2.P1.x) * (L2.P1.y - L2.P2.y) -
        (L1.P1.y - L2.P1.y) * (L2.P1.x - L2.P2.x)
    un =
        (L1.P1.x - L2.P1.x) * (L1.P1.y - L1.P2.y) -
        (L1.P1.y - L2.P1.y) * (L1.P1.x - L1.P2.x)
    D =
        (L1.P1.x - L1.P2.x) * (L2.P1.y - L2.P2.y) -
        (L1.P1.y - L1.P2.y) * (L2.P1.x - L2.P2.x)
    println(L1, L2, D)
    t, u = tn / D, un / D

    if 0 ≤ tn ≤ D && 0 ≤ un ≤ D
        x_cross, y_cross =
            L1.P1.x + t * (L1.P2.x - L1.P1.x), L1.P1.y + t * (L1.P2.y - L1.P1.y)
        println(tn, " ", un, " ", x_cross, " ", y_cross)
        return is_integer(x_cross) && is_integer(y_cross)
    end
    false
end

is_horz(s::LineSegment) = s.P1.y == s.P2.y
is_vert(s::LineSegment) = s.P1.x == s.P2.x

function horz_vert_cross(LS1::LineSegment, LS2::LineSegment)::Int
    if is_horz(LS1) && is_vert(LS2)
        cross = LS1.P1.x ≤ LS2.P1.x ≤ LS1.P2.x && LS2.P1.y ≤ LS1.P1.y ≤ LS2.P2.y
        cross
    elseif is_vert(LS1) && is_horz(LS2)
        cross = LS2.P1.x ≤ LS1.P1.x ≤ LS2.P2.x && LS1.P1.y ≤ LS2.P1.y ≤ LS1.P2.y
        cross
    elseif is_horz(LS1) && is_horz(LS2)
        if LS1.P1.y != LS2.P2.y
            return 0
        end
        length(union(Set(LS1.P1.x:LS1.P2.x), Set(LS2.P1.x:LS2.P2.x)))
    else
        if LS1.P1.x != LS2.P2.x
            return 0
        end
        length(union(Set(LS1.P1.y:LS1.P2.y), Set(LS2.P1.y:LS2.P2.y)))
    end
end

function p1()
    data = filter(LS -> LS.P1.x == LS.P2.x || LS.P1.y == LS.P2.y, get_d5_data())

    crossings = 0
    current = Set()
    for LS in data
        for viewed_LS in current
            crossings += horz_vert_cross(LS, viewed_LS)
            if viewed_LS.P2.x < LS.P1.x
                delete!(current, viewed_LS)
            end
        end
        if is_horz(LS)
            push!(current, LS)
        end
    end

    crossings
end

p1()
# data = filter(LS -> LS.P1.x == LS.P2.x || LS.P1.y == LS.P2.y, get_d5_data())
```




    16



## Day 6!


```julia
using Memoize

function get_d6_data()
    inp_str = open(io -> read(io, String), "inputs/d06.txt", "r") |> strip
    map(s -> parse(Int, s), split(inp_str, ","))
end

@memoize function lanternfish(internal_fish_timer::Int, n_days_left::Int)::Int
    """
    memoizing this would really make it a lot faster (maybe? how often are (IFT, NDL) pairs
    showing up?
    """
    if internal_fish_timer >= n_days_left
        return 1
    end
    lanternfish(8, n_days_left - internal_fish_timer - 1) +
    lanternfish(6, n_days_left - internal_fish_timer - 1)
end

function p1()
    # Internal Fish Timer ==> IFT
    input = get_d6_data()
    mapreduce(IFT -> lanternfish(IFT, 80), +, input)
end

function p2()
    input = get_d6_data()
    mapreduce(IFT -> lanternfish(IFT, 256), +, input)
end
p1(), p2()
```




    (386536, 1732821262171)



Using memoization to make things quicker. Out of interest, here is the size of our cache:


```julia
memoize_cache(lanternfish)
```




    IdDict{Any, Any} with 516 entries:
      (6, 96)  => 3612
      (6, 171) => 2395409
      (8, 154) => 460699
      (8, 227) => 268920395
      (6, 55)  => 106
      (8, 166) => 1296477
      (6, 116) => 19600
      (6, 9)   => 2
      (8, 242) => 990284884
      (6, 86)  => 1421
      (8, 53)  => 70
      (6, 216) => 122267142
      (6, 23)  => 7
      (8, 16)  => 3
      (8, 83)  => 950
      (8, 17)  => 3
      (6, 109) => 10599
      (8, 200) => 25247007
      (6, 122) => 34255
      (6, 162) => 1098932
      (6, 154) => 556666
      (8, 205) => 39025282
      (6, 140) => 166401
      (6, 65)  => 236
      (8, 5)   => 1
      ⋮        => ⋮



Not too bad. Fairly small, considering the exponential growth. A tremendous amount of repeating occurs, making this a prime usecase for a cache.

## Day 7

Our problem is
$$
    C(x;\mathbf{p}) = \sum_{i=0}^N |p_i - x|,  \quad MFC =\min_x C(x;\mathbf{p}), \quad x, p_i \in \mathbb{Z}
$$
where $\mathbf{p}$ is a vector of the current positions $p_i$ of the Crab Submarines, $C(x)$ is the fuel cost for target position $x$, and $MFC$ is the minimum fuel cost possible.

but...

Ignore mathin' and just do some map reducin'


```julia
function get_d7_data()
    inp_str = open(io -> read(io, String), "inputs/d07.txt", "r") |> strip
    map(s -> parse(Int, s), split(inp_str, ","))
end

crab_submarine_positions = get_d7_data()

min_cost(p::Vector{Int}, C::Function) = minimum(map(x -> C(x, p), minimum(p):maximum(p)))

function p1()
    C(x::Int, p::Vector{Int}) = sum(abs.(x .- p))
    min_cost(crab_submarine_positions, C)
end


function p2()
    function C(x::Int, p::Vector{Int})
        N = abs.(x .- p)
        sum(N .* (N .+ 1)) / 2
    end
    min_cost(crab_submarine_positions, C)
end


p1(), p2()
```




    (352254, 9.9053143e7)


