#! /usr/local/bin/julia

using LinearAlgebra
using Combinatorics

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

function get_d8_data()
  data = process_inputs("08test") do s
    signal_patterns, outputs = split(s, " | ")
    sort(split(signal_patterns, " "), by=length), split(outputs, " ")
  end
  [s for (s, _) in data], [Set(o) for (_, o) in data]
end

function better_symdiff(s...)
  common = intersect(s...)
  mapreduce(ss -> setdiff(ss, common), union, s)
end

org_mapping = Dict(
                   0 => Set("ABCEFG"),
                   1 => Set("CF"),
                   2 => Set("ACDEG"),
                   3 => Set("ACDFG"),
                   4 => Set("BCDF"),
                   5 => Set("ABDFG"),
                   6 => Set("ABDEFG"),
                   7 => Set("ACF"),
                   8 => Set("ABCDEFG"),
                   9 => Set("ABCDFG")
                  )

letter_to_num = Dict(
                     'A' => Set([0, 5, 6, 7, 2, 9, 8, 3]),
                     'B' => Set([0, 4, 5, 6, 9, 8]),
                     'C' => Set([0, 4, 7, 2, 9, 8, 3, 1]),
                     'D' => Set([5, 4, 6, 2, 9, 8, 3]),
                     'E' => Set([0, 6, 2, 8]),
                     'F' => Set([0, 4, 5, 6, 7, 9, 8, 3, 1]),
                     'G' => Set([0, 5, 6, 2, 9, 8, 3])
                    )


function deduce_mapping(signal_pattern::Vector{SubString{String}})
  """
  we have MX = T

  M is the 10x7 signal pattern matrix
  X is a 7x7 conversion matrix
  T is the 10x7 target matrix (i.e. correct 7 seg assignment)
  """
  T = [
       0 0 1 0 0 1 0;
       1 0 1 0 0 1 0;
       0 1 1 1 0 1 0;
       1 0 1 1 1 0 1;
       1 0 1 1 0 1 1;
       1 1 0 1 0 1 1;
       1 1 1 1 0 1 1;
       1 1 0 1 1 1 1;
       1 1 1 0 1 1 1;
       1 1 1 1 1 1 1
      ]

  M = zeros(10,7)
  wires = "abcdefg"
  for (i, num_map) in enumerate(signal_pattern)
    for letter in num_map
      M[i, findfirst(letter, wires)] = 1
    end
  end
  M_org = copy(M)

  found = false
  local conv_mat::Matrix
  for p_len_6 in permutations((7,8,9))
    M[[7,8,9],:] = M[p_len_6,:]
    for p_len_5 in permutations((4,5,6))
      M[[4,5,6],:] = M[p_len_5,:]
      conv_mat = pinv(M) * T
      conv_mat = round.(conv_mat, digits=2)
      foreach(f -> println("\t $f"), eachrow(conv_mat))
      println()
      if all(el -> el == 0 || el == 1, conv_mat)
        found = true
        break
      end
    end
    M = copy(M_org)
  end

  if !found
    error("fuck!")
  end

  conv_mat
end

function p1()
  _, outputs = get_d8_data()
  mapreduce(s -> length(s) in [2, 3, 4, 7], +, reduce(vcat, outputs))
end

function p2()
  signal_patterns, outputs = get_d8_data()

  signal_pattern = first(signal_patterns)
  println(deduce_mapping(signal_pattern))
end

p1()
p2()
