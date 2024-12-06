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

	def explore(y, x, grid):
		d = 0
		path = {(y, x, d)}
		while True:
			if not 0 <= y + yd[d] < ymax or not 0 <= x + xd[d] < xmax:
				return len({(y, x) for y, x, _ in path})
			if grid[y + yd[d]][x + xd[d]] == '.':
				y += yd[d]
				x += xd[d]
				if (y, x, d) in path:
					return -1
				path.add((y, x, d))
			else:
				d = (d + 1) % 4

	def grids():
		for y, line in enumerate(grid):
			for x, char in enumerate(line):
				if char == '.':
					yield [line if yy != y
							else [c if xx != x else 'O'
								for xx, c in enumerate(line)]
							for yy, line in enumerate(grid)]

	result1 = explore(y, x, grid)
	result2 = sum(explore(y, x, grid) == -1 for grid in grids())
	return result1, result2


if __name__ == '__main__':
	main(globals())
