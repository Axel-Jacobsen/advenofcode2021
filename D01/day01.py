#! /usr/bin/env python3

with open('input.txt') as f:
    data = [int(s) for s in f.read().split("\n") if s != ""]

def quantify(d, pred=bool):
    return sum(pred(dd) for dd in d)

print(quantify(zip(data,data[1:]), lambda x: x[1]>x[0]))
print(quantify(
    zip(
        zip(data,data[1:],data[2:]),
        zip(data[1:],data[2:],data[3:])
    ), lambda x: sum(x[1])>sum(x[0])))
