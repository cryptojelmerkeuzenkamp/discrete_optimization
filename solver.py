#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import List

from collections import namedtuple
from algorithms import simple_fill
from branch_and_bound import BranchAndBoundSolver
Item = namedtuple("Item", ['index', 'value', 'weight'])


def _parse_input(input_data: str):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    return capacity, items


def _parse_output(value: int, taken: List):
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def solve_it(input_data: str):
    """
    Modify this code to run your optimization algorithm.
    :param input_data:
    :return:
    """
    # Parse input
    capacity, items = _parse_input(input_data)

    # Branch and Bound
    value, taken = BranchAndBoundSolver(items, capacity).execute()

    # Parse output
    output_data = _parse_output(value, taken)
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

