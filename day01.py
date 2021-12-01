#! /usr/bin/env python3

from helpers import *


data = process_inputs("01", int)

print(quantify(zip(data, data[1:]), lambda x: x[1] > x[0]))
print(
    quantify(
        zip(zip(data, data[1:], data[2:]), zip(data[1:], data[2:], data[3:])),
        lambda x: sum(x[1]) > sum(x[0]),
    )
)
