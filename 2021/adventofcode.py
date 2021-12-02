"""Advent of Code 2021. http://adventofcode.com/2021 """
import os
import re
import sys
# import operator
# import itertools
# from functools import reduce
# from collections import Counter, defaultdict
# import numpy as np
# from numba import njit


def day1a(s):
	nums = [int(a) for a in s.splitlines()]
	return sum(a < b for a, b in zip(nums, nums[1:]))


def day1b(s):
	nums = [int(a) for a in s.splitlines()]
	sums = [a + b + c for a, b, c in zip(nums, nums[1:], nums[2:])]
	return sum(a < b for a, b in zip(sums, sums[1:]))


def day2a(s):
	horiz = depth = 0
	for line in s.splitlines():
		op, operand = line.split()
		if op == 'down':
			depth += int(operand)
		elif op == 'up':
			depth -= int(operand)
		elif op == 'forward':
			horiz += int(operand)
	return horiz * depth


def day2b(s):
	horiz = depth = aim = 0
	for line in s.splitlines():
		op, operand = line.split()
		if op == 'down':
			aim += int(operand)
		elif op == 'up':
			aim -= int(operand)
		elif op == 'forward':
			horiz += int(operand)
			depth += aim * int(operand)
	return horiz * depth


def benchmark():
	import timeit
	for name in list(globals()):
		match = re.match(r'day(\d+)[ab]', name)
		if match is not None and os.path.exists('i%s' % match.group(1)):
			time = timeit.timeit(
					'%s(inp)' % name,
					setup='inp = open("i%s").read().rstrip("\\n")'
						% match.group(1),
					number=1,
					globals=globals())
			print('%s\t%5.2fs' % (name, time))


def main():
	if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
		benchmark()
	elif len(sys.argv) > 1 and sys.argv[1].startswith('day'):
		with open('i' + sys.argv[1][3:].rstrip('ab') if len(sys.argv) == 2
				else sys.argv[2]) as inp:
			print(globals()[sys.argv[1]](inp.read().rstrip('\n')))
	else:
		raise ValueError('unrecognized command. '
				'usage: python3 adventofcode.py day[1-25][ab] [input]'
				'or: python3 adventofcode.py benchmark')


if __name__ == '__main__':
	main()
