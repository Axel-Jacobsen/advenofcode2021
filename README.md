# Advent of Code, 2021!

It is truly the most wonderful time of the year! Here is my work for this year's Advent.

I am doing it in Julia, Rust (slowly), and hopefully another, more functional language for some problems.

Folders hold solutions in languages other than Julia. This is the Julia notebook. Enjoy!


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
```




    process_inputs (generic function with 2 methods)



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

p1(), p2()
```




    (1499229, 1340836560)



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


```julia
function get_d5_data()::Vector{LineSegment}
    data = process_inputs("05") do s
        r1_str, r2_str = split(s, " -> ")
        x1, y1 = map(v -> parse(Int64, v), split(r1_str, ","))
        x2, y2 = map(v -> parse(Int64, v), split(r2_str, ","))
        x1 ≤ x2 ? LineSegment(Point(x1, y1), Point(x2, y2)) :
        LineSegment(Point(x2, y2), Point(x1, y1))
    end
    sort(data, by = LS -> LS.P1.x)
end

struct Point{T<:Integer}
    x::T
    y::T
end

struct LineSegment{T<:Integer}
    P1::Point{T}
    P2::Point{T}
end

is_horz(LS::LineSegment) = LS.P1.y == LS.P2.y
is_vert(LS::LineSegment) = LS.P1.x == LS.P2.x

function get_points(LS::LineSegment)::Base.Iterators.Zip
    step = LS.P1.y ≤ LS.P2.y ? 1 : -1
    if is_horz(LS)
        zip(LS.P1.x:LS.P2.x, [LS.P1.y for _ = LS.P1.x:LS.P2.x])
    elseif is_vert(LS)
        zip([LS.P1.x for _ = LS.P1.y:step:LS.P2.y], LS.P1.y:step:LS.P2.y)
    else
        zip(LS.P1.x:LS.P2.x, LS.P1.y:step:LS.P2.y)
    end
end

function crosses(LS1::LineSegment, LS2::LineSegment)::Set{Tuple{Int,Int}}
    # optimization: check if LS1 and LS2 intersect at 1 point or multiple
    # if 1 point, can speed this up quite a bit
    # otherwise, this is fast enough for now
    LS1_points = get_points(LS1)
    LS2_points = get_points(LS2)
    Set(intersect(LS1_points, LS2_points))
end

function p(data)
    crossings = Set{Tuple{Int,Int}}()
    current = Set{LineSegment}()
    for LS in data
        for viewed_LS in current
            if viewed_LS.P2.x < LS.P1.x
                delete!(current, viewed_LS)
            else
                crossing_points = crosses(LS, viewed_LS)
                union!(crossings, crossing_points)
            end
        end
        push!(current, LS)
    end
    length(crossings)
end

p1_data = filter(LS -> LS.P1.x == LS.P2.x || LS.P1.y == LS.P2.y, get_d5_data())
p2_data = get_d5_data()

p(p1_data), p(p2_data)
```




    (4873, 19472)



## Day 6!


```julia
using Memoize

function get_d6_data()
    inp_str = open(io -> read(io, String), "inputs/d06.txt", "r") |> strip
    map(s -> parse(Int, s), split(inp_str, ","))
end

@memoize function lanternfish(internal_fish_timer::Int, n_days_left::Int)::Int
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

## Day 7!

Our problem is
$$
    C(x;\mathbf{p}) = \sum_{i=0}^N |p_i - x|,  \quad MFC =\min_x C(x;\mathbf{p}), \quad x, p_i \in \mathbb{Z}
$$
where $\mathbf{p}$ is a vector of the current positions $p_i$ of the Crab Submarines, $C(x)$ is the fuel cost for target position $x$, and $MFC$ is the minimum fuel cost possible.

but...

Ignore mathin' and just do some map reducin'. There is probably a cleaner way to do this that relies on math, but since the system is nonlinear (due to the $\mid \cdot \mid$), it is tricky, and I am not aware of the method. The map is pretty quick.


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



## Day 8!


```julia
function get_d8_data()
    data = process_inputs("08") do s
        signal_patterns, outputs = split(s, " | ")
        sort(split(signal_patterns, " "), by = length), split(outputs, " ")
    end
end

org_mapping = Dict(
    Set("abcefg") => 0,
    Set("cf") => 1,
    Set("acdeg") => 2,
    Set("acdfg") => 3,
    Set("bcdf") => 4,
    Set("abdfg") => 5,
    Set("abdefg") => 6,
    Set("acf") => 7,
    Set("abcdefg") => 8,
    Set("abcdfg") => 9,
)

function deduce_mapping(signal_pattern::Vector)
    one, seven, four, eight =
        signal_pattern[1], signal_pattern[2], signal_pattern[3], signal_pattern[end]
    a = setdiff(seven, one)
    nine_zero_six = filter(s -> length(s) == 6, signal_pattern)
    dce = mapreduce(num -> setdiff(eight, num), union, nine_zero_six)
    c = intersect(dce, one)
    f = setdiff(seven, a, c)
    d = setdiff(intersect(dce, four), c)
    e = setdiff(dce, c, d)
    g = setdiff(eight, four, seven, e)
    b = setdiff(eight, a, c, d, e, f, g)
    Dict(zip(map(pop!, (a, b, c, d, e, f, g)), "abcdefg"))
end

function p1()
    outputs = [o for (_, o) in get_d8_data()]
    mapreduce(s -> length(s) in [2, 3, 4, 7], +, reduce(vcat, outputs))
end

function p2()
    sigs_outs = get_d8_data()

    s = 0
    for (signal_pattern, output) in sigs_outs
        pairs = deduce_mapping(signal_pattern)
        for (i, digit) in enumerate(output)
            num = org_mapping[Set(map(s -> pairs[s], digit))]
            s += num * 10^(4 - i)
        end
    end
    s
end

p1(), p2()
```




    (554, 990964)



## Day 9!


```julia
function d09data()
    process_inputs("09") do line
        # vector of ints
        map(s -> parse(Int, s), collect(line))
        # convert vec of vec of int to matrix of int
    end |> vv -> mapreduce(permutedims, vcat, vv)
end

function p1()
    data = d09data()

    total_hgt = 0
    max_i, max_j = size(data)
    for i = 1:max_i
        for j = 1:max_j
            ctr = data[i, j]
            abv = i - 1 < 1 ? 9 : data[i-1, j]
            lft = j - 1 < 1 ? 9 : data[i, j-1]
            blw = i + 1 > max_i ? 9 : data[i+1, j]
            rgt = j + 1 > max_j ? 9 : data[i, j+1]
            is_lowpoint = all(pt -> ctr < pt, (abv, blw, rgt, lft))
            total_hgt += is_lowpoint ? 1 + ctr : 0
        end
    end
    total_hgt
end

Base.:+(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}) = (p1[1] + p2[1], p1[2] + p2[2])
Base.:≤(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}) = p1[1] ≤ p2[1] && p1[2] ≤ p2[2]

function p2()
    D = d09data()

    visited_points = Set{Tuple{Int,Int}}()
    function spread(D, i::Int, j::Int)
        points_to_visit = Set{Tuple{Int,Int}}([(i, j)])

        count = 0
        while length(points_to_visit) > 0
            point = pop!(points_to_visit)

            if D[point...] == 9
                push!(visited_points, point)
            elseif !(point in visited_points)
                count += 1
                push!(visited_points, point)
                next_points = [
                    point + dir for dir in ((1, 0), (-1, 0), (0, 1), (0, -1)) if
                    (1, 1) ≤ point + dir ≤ size(D) && !(point + dir in visited_points)
                ]
                if length(next_points) > 0
                    push!(points_to_visit, next_points...)
                end
            end
        end
        count
    end

    vals = []
    max_i, max_j = size(D)
    for (i, j) in Iterators.product(1:max_i, 1:max_j)
        if !((i, j) in visited_points)
            push!(vals, spread(D, i, j))
        end
    end
    prod(sort!(vals)[end-2:end])
end

p1(), p2()
```




    (491, 1075536)



## Day 10!


```julia
data = process_inputs(collect, "10")

paren_pairs = Dict('(' => ')', '[' => ']', '{' => '}', '<' => '>')

function gen_stack(line::Vector{Char})::Vector{Char}
    """ if corrupt, return the corrupting char, else nothing
    """
    stk = [line[1]]
    for char in line[2:end]
        if char in (')', '>', ']', '}')
            if paren_pairs[stk[end]] == char
                pop!(stk)
            else
                push!(stk, char)
                return stk
            end
        else
            push!(stk, char)
        end
    end
    stk
end

function p1()
    conversion = Dict(')' => 3, ']' => 57, '}' => 1197, '>' => 25137)
    syntax_err_score = 0
    mapreduce(+, data) do line
        stk = gen_stack(line)
        stk[end] in (')', '>', ']', '}') ? conversion[stk[end]] : 0
    end
end


function p2()
    conversion = Dict(')' => 1, ']' => 2, '}' => 3, '>' => 4)
    scores = []
    for line in data
        stk = gen_stack(line)
        if !(stk[end] in (')', '>', ']', '}'))
            score = 0
            while length(stk) > 0
                score *= 5
                score += conversion[paren_pairs[pop!(stk)]]
            end
            push!(scores, score)
        end
    end
    sort(scores)[length(scores)÷2+1]
end

p1(), p2()
```




    (316851, 2182912364)


