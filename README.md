# Advent of Code, 2021!

It is truly the most wonderful time of the year! Here is my work for this year's Advent.


```julia
using DataStructures
using JupyterFormatter
enable_autoformat()

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

function matrix_inputs(day::String)
    process_inputs(day) do line
        # vector of ints
        map(s -> parse(Int, s), collect(line))
        # convert vec of vec of int to matrix of int
    end |> vv -> mapreduce(permutedims, vcat, vv)
end

clockmod(i::Int, m::Int) = i % m == 0 ? m : i % m

cartdirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
diagdirs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (-1, -1), (1, -1))

Base.:+(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}) = (p1[1] + p2[1], p1[2] + p2[2])
Base.:≤(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}) = p1[1] ≤ p2[1] && p1[2] ≤ p2[2]
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
function p1()
    data = matrix_inputs("09")

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

function p2()
    D = matrix_inputs("09")
    visited_points = Set{Tuple{Int,Int}}()

    function spread(D, i::Int, j::Int)
        count = 0
        points_to_visit = Set{Tuple{Int,Int}}([(i, j)])
        while length(points_to_visit) > 0
            point = pop!(points_to_visit)
            if D[point...] == 9
                push!(visited_points, point)
            elseif !(point in visited_points)
                count += 1
                push!(visited_points, point)
                next_points = [
                    point + dir for dir in cartdirs if
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



## Day 11!


```julia
function octopus_step!(octopi)
    """ At each step
    ∀ (i,j) D_ij += 1
    while ∃ D_ij > 9
        ∀ D_ij > 9, D_{i ± 1},{j ± 1} += 1
    D_ij = 0 ∀ D_i,j > 9 
    """
    flashed_idxs = Set{Tuple{Int,Int}}()
    octopi .+= ones(size(octopi))

    flashing_octopi = Tuple.(findall(x -> x > 9, octopi))
    while length(flashing_octopi) > 0
        for octidx in flashing_octopi
            foreach(
                dir -> octopi[octidx + dir...] += 1,
                [dir for dir in diagdirs if (1, 1) ≤ octidx + dir ≤ size(octopi)],
            )
        end

        push!(flashed_idxs, flashing_octopi...)  # add the new oc
        flashing_octopi = Tuple.(findall(x -> x > 9, octopi))  # get all octopi w/ energy > 9
        setdiff!(flashing_octopi, flashed_idxs)  # remove previous flashes from this step
    end

    if length(flashed_idxs) > 0
        foreach(flashed_idx -> octopi[flashed_idx...] = 0, flashed_idxs)
    end
    length(flashed_idxs)
end

function p1()
    octopi = matrix_inputs("11")

    total_flashes = 0
    for i = 1:100
        total_flashes += octopus_step!(octopi)
    end
    total_flashes
end

function p2()
    octopi = matrix_inputs("11")

    i = 1
    while true
        num_flashed_octopi = octopus_step!(octopi)
        if num_flashed_octopi == 100
            return i
        end
        i += 1
    end
end

p1(), p2()
```




    (1686, 360)



## Day 12!


```julia
function build_graph()
    graph = Dict{String,Set{String}}()
    process_inputs("12") do line
        n1, n2 = split(line, "-")
        haskey(graph, n1) ? push!(graph[n1], n2) : graph[n1] = Set([n2])
        haskey(graph, n2) ? push!(graph[n2], n1) : graph[n2] = Set([n1])
    end
    graph
end

function dfs2(graph::Dict{String,Set{String}}, can_visit::Function)::Vector{Vector{String}}
    function dfs_recur(node::String, visited::Dict{String,Int})::Vector{Vector{String}}
        visited[node] += 1
        if node == "end"
            return [["end"]]
        end
        paths = Vector{Vector{String}}()
        for con in graph[node]
            if can_visit(visited, node)
                append!(
                    paths,
                    [
                        [node, rest_of_path...] for
                        rest_of_path in dfs_recur(con, copy(visited))
                    ],
                )
            end
        end
        paths
    end
    visited = Dict([c => 0 for c in keys(graph)])
    dfs_recur("start", visited)
end

function p1()
    graph = build_graph()
    length(
        dfs2(graph, (visited, node) -> all(islowercase, node) ? visited[node] < 2 : true),
    )
end

function p2()
    graph = build_graph()
    repeatable_nodes =
        filter(n -> all(islowercase, n) && !(n in ["start", "end"]), collect(keys(graph)))

    all_paths = Set{Vector{String}}()
    for repeat_node in repeatable_nodes
        function can_visit(visited, node)
            if node == repeat_node
                visited[node] < 3
            else
                all(islowercase, node) ? visited[node] < 2 : true
            end
        end

        for p in dfs2(graph, can_visit)
            push!(all_paths, p)
        end
    end
    length(all_paths)
end

p1(), p2()
```




    (4241, 122134)



## Day 13!


```julia
function get_d13_data()
    points = Set{Tuple{Int,Int}}()
    folds = Vector{Tuple{String,Int}}()

    open("inputs/d13.txt", "r") do io
        for line in eachline(io)
            # We start with a bunch of points
            nums = match(r"(\d+),(\d+)", line)
            if nums != nothing
                point = (parse(Int64, nums[1]), parse(Int64, nums[2]))
                push!(points, point)
            elseif match(r"^\s*$", line) == nothing
                fold_instr = split(replace(line, "fold along " => ""), "=")
                push!(folds, (fold_instr[1], parse(Int64, fold_instr[2])))
            end
        end
    end
    points, folds
end

function applyfold!(points::Set{Tuple{Int,Int}}, fold::Tuple{String,Int})
    """ apply fold along fold[1] (either "x" or "y") = fold[2]
    """
    fold_idx = fold[1] == "x" ? 1 : 2
    fold_pos = fold[2]

    # foreach seems to make a 
    foreach(points) do point
        if point[fold_idx] > fold_pos
            delete!(points, point)
            if fold[1] == "x"
                folded_point = (2fold_pos - point[fold_idx], point[2])
            else
                folded_point = (point[1], 2fold_pos - point[fold_idx])
            end
            push!(points, folded_point)
        end
    end
end

function printpoints(points::Set{Tuple{Int,Int}})
    min_x, max_x = minimum(p -> p[1], points), maximum(p -> p[1], points)
    min_y, max_y = minimum(p -> p[2], points), maximum(p -> p[2], points)

    for j = min_y:max_y
        for i = min_x:max_x
            if (i, j) in points
                print('#')
            else
                print(' ')
            end
        end
        println()
    end
end

function p1()
    points, folds = get_d13_data()
    applyfold!(points, folds[1])
    length(points)
end

function p2()
    points, folds = get_d13_data()
    foreach(fold -> applyfold!(points, fold), folds)
    printpoints(points)
end

p1(), p2()
```

    #### ###  #     ##  ###  #  # #    ### 
    #    #  # #    #  # #  # #  # #    #  #
    ###  #  # #    #    #  # #  # #    #  #
    #    ###  #    # ## ###  #  # #    ### 
    #    #    #    #  # # #  #  # #    # # 
    #### #    ####  ### #  #  ##  #### #  #





    (710, nothing)



## Day 14!


```julia
function get_d14_data()
    conversions = Dict{Tuple{Char,Char},Char}()
    open("inputs/d14.txt", "r") do io
        template = readline(io)
        readline(io) # skip empty line
        for line in eachline(io)
            fm, to = split(line, " -> ")
            conversions[Tuple(collect(fm))] = only(to)
        end
        collect(template), conversions
    end
end

defaultinc!(d::Dict, k::Any) = k in keys(d) ? d[k] += 1 : d[k] = 1
defaultinc!(d::Dict, k::Any, v::Int) = k in keys(d) ? d[k] += v : d[k] = v

function step_smartly!(
    polymer::Dict{Tuple{Char,Char},Int},
    conversions::Dict{Tuple{Char,Char},Char},
)
    pp = copy(polymer)
    for (pair, count) in polymer
        if pair in keys(conversions) && count > 0
            defaultinc!(pp, (pair[1], conversions[pair]), count)
            defaultinc!(pp, (conversions[pair], pair[2]), count)
            pp[pair] -= count
        end
    end
    foreach(((k, v),) -> polymer[k] = v, pp)
end

function p(num_steps)
    template, conversions = get_d14_data()
    polymer = Dict{Tuple{Char,Char},Int}()

    for pair in zip(template, template[2:end])
        defaultinc!(polymer, pair)
    end

    for _ = 1:num_steps
        step_smartly!(polymer, conversions)
    end

    # get char counts
    charcounts = Dict{Char,Int}()
    for (pair, count) in polymer
        defaultinc!(charcounts, pair[1], count)
        defaultinc!(charcounts, pair[2], count)
    end

    startchar, endchar = template[1], template[end]
    counts = [
        char in [startchar, endchar] ? (count + 1) / 2 : count / 2 for
        (char, count) in charcounts
    ]
    maximum(counts) - minimum(counts)
end

p(10), p(40)
```




    (3213.0, 3.711743744429e12)



## Day 15!


```julia
M = matrix_inputs("15")

struct RepeatingGrid{T} <: AbstractArray{T,2}
    mat::Matrix{T}
    repeat::Int
    RepeatingGrid(m::Matrix{T}) where {T} = new{T}(m, 5)
    RepeatingGrid(m::Matrix{T}, r::Int) where {T} = new{T}(m, r)
end

Base.size(m::RepeatingGrid) = (size(m.mat)[1] * m.repeat, size(m.mat)[2] * m.repeat)
Base.IndexStyle(::RepeatingGrid) = IndexCartesian()
Base.firstindex(m::RepeatingGrid) = (1, 1)
Base.lastindex(m::RepeatingGrid) = size(m)

function Base.getindex(m::RepeatingGrid, I::Vararg{Int,2})
    i, j = I
    matrix_i, matrix_j = size(m.mat)
    if i > matrix_i * m.repeat || j > matrix_j * m.repeat
        throw(BoundsError(m, i, j))
    end
    increment = ((i - 1) ÷ matrix_i) + ((j - 1) ÷ matrix_j)
    return clockmod(m.mat[clockmod(i, matrix_i), clockmod(j, matrix_i)] + increment, 9)
end

function astar(M::AbstractArray{Int,2}, startpt::Tuple{Int,Int}, endpt::Tuple{Int,Int})
    h(x::Tuple{Int,Int})::Int = abs(endpt[1] - x[1]) + abs(endpt[2] - x[2])
    Q = PriorityQueue{Tuple{Int,Int},Int}([startpt => 0])
    risklength = Dict{Tuple{Int,Int},Int}([startpt => 0])

    while !isempty(Q)
        currentnode = dequeue!(Q)
        if currentnode == endpt
            break
        end

        neighbors =
            filter(n -> ((1, 1) ≤ n ≤ size(M)), [currentnode + dir for dir in cartdirs])
        for neighbor in neighbors
            neighborrisk = risklength[currentnode] + M[neighbor...]
            if !(neighbor in keys(risklength)) || neighborrisk < risklength[neighbor]
                risklength[neighbor] = neighborrisk
                Q[neighbor] = neighborrisk + h(neighbor)
            end
        end
    end
    risklength[endpt...]
end

function p1()
    astar(M, (1, 1), size(M))
end

function p2()
    G = RepeatingGrid(M)
    astar(G, (1, 1), size(G))
end

p1(), p2()
```




    (714, 2948)



## Day 16!


```julia
mutable struct BITS
    bits::String
    idx::Int
    BITS(bits::String) = new(bits, 1)
end

function Base.iterate(B::BITS, state::Int = 1)
    state = B.idx
    if state > length(B)
        return nothing
    end
    letteridx = ((state - 1) ÷ 4) + 1
    bitidx = 5 - clockmod(state, 4)
    res = (parse(Int, B.bits[letteridx], base = 16) & (1 << (bitidx - 1))) >> (bitidx - 1)
    B.idx = state + 1
    (res, B.idx)
end

Base.length(B::BITS) = length(B.bits) * 4

binarrtoint(binarr) = parse(Int, join(binarr), base = 2)

# really just cause Iterators.take is long to type
bite(B::BITS, n::Int, post = binarrtoint) = Iterators.take(B, n) |> post

function process_packet_values(pkt::BITS)
    vs = Int[]
    while true
        chunk = bite(pkt, 5, collect)
        push!(vs, chunk[2:end]...)
        if chunk[1] == 0
            break
        end
    end
    vs
end

function process_packet_p1(pkt::BITS)
    version, typeid = bite(pkt, 3), bite(pkt, 3)
    if typeid == 4
        process_packet_values(pkt)
        return version
    end
    s = 0
    lengthTypeId = bite(pkt, 1)
    if lengthTypeId == 0
        length_subpackets = bite(pkt, 15)
        start_len = pkt.idx
        while pkt.idx < start_len + length_subpackets
            s += process_packet_p1(pkt)
        end
    else
        num_subpackets = bite(pkt, 11)
        for _ = 1:num_subpackets
            s += process_packet_p1(pkt)
        end
    end
    s + version
end

function process_packet(pkt::BITS)
    gtInt(a, b) = Int(a > b)
    ltInt(a, b) = Int(a < b)
    ops = Dict([0 => +, 1 => *, 2 => min, 3 => max, 5 => gtInt, 6 => ltInt, 7 => ==])

    version, typeid = bite(pkt, 3), bite(pkt, 3)
    if typeid == 4
        # chunks of 5 bits starting w/ 1, and then the last starts w/ 0
        return process_packet_values(pkt) |> binarrtoint
    end
    vs = Int[]
    lengthTypeId = bite(pkt, 1)
    if lengthTypeId == 0
        length_subpackets = bite(pkt, 15)
        start_len = pkt.idx
        while pkt.idx < start_len + length_subpackets
            push!(vs, process_packet(pkt))
        end
    else
        num_subpackets = bite(pkt, 11)
        for _ = 1:num_subpackets
            push!(vs, process_packet(pkt))
        end
    end
    ops[typeid](vs...)
end

function p1()
    B = process_inputs(s -> BITS(s), "16")[1]  # id func to just get the string
    process_packet_p1(B)
end

function p2()
    B = process_inputs(s -> BITS(s), "16")[1]
    process_packet(B)
end

p1(), p2()
```




    (1014, 1922490999789)



## Day 17!


```julia
xmin, xmax, ymin, ymax =
    process_inputs("17test") do line
        match(r"x=(-?\d+)..(-?\d+), y=(-?\d+)..(-?\d+)", line).captures |>
        s -> map(v -> parse(Int, v), s)
    end |> pop!

# you can calculate simple equation of motion for y, and re-arrange:
# V0y = y / n - a * (n - 1) / 2
V0y(y::Int, n::Int; a::Int = -1) = y / n - a * (n - 1) / 2

# take integer solutions for V0y
permissible_V0ys(y::Int; a = -1) =
    filter(isinteger, V0y.(y, -abs(y):abs(y), a = a)) |> unique

# scan through all ys for all permissible solutions 
all_permissible_V0ys = reduce(vcat, permissible_V0ys.(ymin:ymax)) |> unique

function p1()
    # Vy = 0 at max height
    maxheight(V0y; a = -1) = V0y <= 0 ? 0 : abs(a) * V0y * (V0y + 1) / 2
    map(maxheight, all_permissible_V0ys) |> maximum
end

permissible_n_V0ys(y::Int; a = -1) =
    filter(p -> isinteger(p[2]), (n -> (n, V0y(y, n, a = a))).(-abs(y):abs(y))) |> unique
all_permissible_n_V0ys = reduce(vcat, permissible_n_V0ys.(ymin:ymax)) |> unique

function p2()
    vs = []
    for (n, V0y) in all_permissible_n_V0ys
        for V0x = 1:xmax
            if xmin ≤ sum([V0x - j for j = 0:n-1]) ≤ xmax
                push!(vs, (V0x, V0y))
            end
        end
    end
    vs |> length
end

p1(), "Part 2 is unfinished"
```




    (45.0, "Part 2 is unfinished")



## Day 18!

this is so tree-like that I am nauseous


```julia

```




    "[[1,2],3]"


