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


def day3a(s):
	lines = s.splitlines()
	threshold = len(lines) // 2
	counts = [0] * len(lines[0])
	for line in lines:
		for n, a in enumerate(line):
			counts[n] += a == '1'
	gamma = ''.join('1' if a >= threshold else '0' for a in counts)
	epsilon = ''.join('1' if a < threshold else '0' for a in counts)
	return int(gamma, base=2) * int(epsilon, base=2)


def day3b(s):
	result = []
	for x in range(2):
		lines = s.splitlines()
		n = 0
		while len(lines) > 1:
			count = sum(int(line[n]) for line in lines)
			threshold = len(lines) // 2 + (len(lines) % 2 != 0)
			bit = (count >= threshold) == (x == 0)
			lines = [line for line in lines if int(line[n]) == bit]
			n += 1
		result.append(int(lines[0], base=2))
	return result[0] * result[1]


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
