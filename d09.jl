#! /usr/local/bin/julia


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


function d09data()
    process_inputs("09test") do line
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

function p2()
    """
    21   43210
    3 878 4 21
     85678 8 2
    87678 678 
     8   65678

    """
    D = d09data()
    spread(D, 1, 1)
end

Base.:+(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}) = (p1[1] + p2[1], p1[2] + p2[2])
Base.:≤(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}) = p1[1] ≤ p2[1] && p1[2] ≤ p2[2]

function spread(D::Matrix{Int}, i::Int, j::Int)
    """
    spread out from point (i,j) in D, stopping
    if there is a 9

    return the entire possible area for a certain D
    from point (i,j)

    probably not computationally efficient, but hey
    neither are u
    """
    println(D)
    visited_points = Set{Tuple{Int,Int}}()
    points_to_visit = Set{Tuple{Int,Int}}([(i, j)])

    while length(points_to_visit) > 0
        point = pop!(points_to_visit)

        if D[point...] == 9 || point in visited_points
            continue
        end

        push!(visited_points, point)
        push!(
            points_to_visit,
            [
                point + dir for
                dir in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (1, 1) ≤ point + dir ≤ size(D)
            ]...,
        )
    end
    length(visited_points)
end

println(p2())
