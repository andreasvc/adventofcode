"""Advent of Code 2024. http://adventofcode.com/2024 """
import re
import sys
from collections import Counter
sys.path.append('..')
from common import main


def day1(s):
	data = [[int(a) for a in line.split()] for line in s.splitlines()]
	a, b = list(map(sorted, zip(*data)))
	cnt = Counter(b)
	result1 = sum(abs(x - y) for x, y in zip(a, b))
	result2 = sum(x * cnt[x] for x in a)
	return result1, result2


def day2(s):
	def safe(data):
		return all(a > b and 1 <= a - b <= 3 for a, b in zip(data, data[1:])
			) or all(a < b and 1 <= b - a <= 3 for a, b in zip(data, data[1:]))

	result1 = result2 = 0
	for line in s.splitlines():
		data = [int(a) for a in line.split()]
		result1 += safe(data)
		result2 += any(safe(data[:n] + data[n + 1:])
				for n in range(len(data)))
	return result1, result2


def day3(s):
	result1 = 0
	for a, b in re.findall(r'mul\(([0-9]{1,3}),([0-9]{1,3})\)', s):
		result1 += int(a) * int(b)
	result2 = 0
	enabled = True
	instr = re.compile(r"(don't\(\)|do\(\)|mul\(([0-9]{1,3}),([0-9]{1,3})\))")
	for op, a, b in instr.findall(s):
		if op == "don't()":
			enabled = False
		elif op == 'do()':
			enabled = True
		elif op.startswith('mul') and enabled:
			result2 += int(a) * int(b)
	return result1, result2


def day4(s, q='XMAS', qq='MAS'):
	result1 = result2 = 0
	data = {(n, m): char
			for n, line in enumerate(s.splitlines())
				for m, char in enumerate(line)}
	for n, m in data:
		# horizontal, vertical, diagonals
		result1 += ''.join(data.get((n + x, m), '')
				for x in range(len(q))) in (q, q[::-1])
		result1 += ''.join(data.get((n, m + x), '')
				for x in range(len(q))) in (q, q[::-1])
		result1 += ''.join(data.get((n + x, m + x), '')
				for x in range(len(q))) in (q, q[::-1])
		result1 += ''.join(data.get((n - x, m + x), '')
				for x in range(len(q))) in (q, q[::-1])
		result2 += ''.join(data.get((n + x, m + x), '')
				for x in range(-1, 2)) in (qq, qq[::-1]) and ''.join(
					data.get((n - x, m + x), '')
				for x in range(-1, 2)) in (qq, qq[::-1])
	return result1, result2


def day5(s):
	from functools import cmp_to_key
	def cmp(a, b):
		if a == b:
			return 0
		if (a, b) in order:
			return -1
		return 1

	order, updates = s.split('\n\n')
	order = {tuple(map(int, a.split('|'))) for a in order.splitlines()}
	key = cmp_to_key(cmp)
	result1 = result2 = 0
	for line in updates.splitlines():
		upd = list(map(int, line.split(',')))
		if all(a not in upd or b not in upd or upd.index(a) < upd.index(b)
				for a, b in order):
			result1 += upd[len(upd) // 2]
		else:
			upd = sorted(upd, key=key)
			result2 += upd[len(upd) // 2]
	return result1, result2


def day6(s):
	grid = s.splitlines()
	ymax, xmax = len(grid), len(grid[0])
	x, y = max((line.find('^'), y) for y, line in enumerate(grid))
	grid[y] = grid[y].replace('^', '.')
	yd, xd = [-1, 0, 1, 0], [0, 1, 0, -1]

	def explore(y, x, d, grid, path=None):
		path = path or {(y, x, d): None}
		while 0 <= y + yd[d] < ymax and 0 <= x + xd[d] < xmax:
			if grid[y + yd[d]][x + xd[d]] == '.':
				y += yd[d]
				x += xd[d]
			else:
				d = (d + 1) % 4
			if (y, x, d) in path:
				return -1
			path[y, x, d] = None
		return path

	def newgrid(y, x):
		return [['#' if xx == x and yy == y else c
				for xx, c in enumerate(line)]
				for yy, line in enumerate(grid)]

	path = list(explore(y, x, 0, grid))
	result1 = len({(y, x) for y, x, _ in path})
	result2 = len({(y2, x2)
			for (n, (y1, x1, d1)), (y2, x2, _) in zip(enumerate(path), path[1:])
			if not any(y3 == y2 and x3 == x2 for y3, x3, _ in path[:n + 1])
			and explore(y1, x1, d1, newgrid(y2, x2), dict.fromkeys(path[:n + 1])) == -1})
	return result1, result2


def day7(s):
	def myeval(nums, ops, outcome):
		result = nums[0]
		for op, num in zip(ops, nums[1:]):
			if op == '+':
				result += num
			elif op == '*':
				result *= num
			elif op == '||':
				result = result * 10 ** int(log10(num) + 1) + num
			else:
				raise ValueError
		return result

	from itertools import product
	from math import log10
	result1 = result2 = 0
	for line in s.splitlines():
		outcome, nums = line.split(':')
		outcome, nums = int(outcome), [int(a) for a in nums.split()]
		for ops in product(['+', '*', '||'], repeat=len(nums) - 1):
			if myeval(nums, ops, outcome) == outcome:
				if '||' not in ops:
					result1 += outcome
				result2 += outcome
				break
	return result1, result2


if __name__ == '__main__':
	main(globals())
