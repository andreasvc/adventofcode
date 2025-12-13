"""Advent of Code 2024. http://adventofcode.com/2024 """
import re
import sys
import math
import itertools
from functools import cache
sys.path.append('..')
from common import main


def day1(s):
	cur = 50
	result1 = result2 = 0
	for line in s.splitlines():
		dir, steps = line[0], int(line[1:])
		prev = cur
		if dir == 'L':
			cur = (cur - steps) % 100
			cond = (prev < cur and prev > 0)
		else:  # dir == 'R':
			cur = (cur + steps) % 100
			cond = prev > cur
		if cur == 0:
			result1 += 1
		if cond or cur == 0:
			result2 += 1
		result2 += steps // 100
	return result1, result2


def day2(s):
	result1 = result2 = 0
	repetition = re.compile(r'([0-9]+)\1+$')
	for rng in s.split(','):
		a, b = map(int, rng.split('-'))
		for n in range(a, b + 1):
			ns = str(n)
			match = repetition.match(ns)
			if match:
				result2 += n
				if ns.count(match.group(1)) == 2:
					result1 += n
	return result1, result2


def day3(s):
	result1 = result2 = 0
	for bank in s.splitlines():
		result1 += max(int(a + b)
				for n, a in enumerate(bank)
					for b in bank[n + 1:])
		jolt = ''
		for n in range(12):
			digit = max(bank[:len(bank) - (12 - n - 1)])
			bank = bank[bank.index(digit) + 1:]
			jolt += digit
		result2 += int(jolt)
	return result1, result2


def day4(s):
	rolls = {(y, x)
			for y, line in enumerate(s.splitlines())
			for x, c in enumerate(line)
			if c == '@'}
	dirs = ((-1, 0), (1, 0), (0, -1), (0, 1),
			(-1, -1), (1, 1), (1, -1), (-1, 1))
	result1 = sum(
			sum((y + dy, x + dx) in rolls
				for dy, dx in dirs)
			< 4
			for y, x in rolls)
	result2 = 0
	while True:
		remove = {
			(y, x) for y, x in rolls
			if sum((y + dy, x + dx) in rolls
				for dy, dx in dirs) < 4}
		if not remove:
			break
		rolls -= remove
		result2 += len(remove)
	return result1, result2


def day5(s):
	ranges, available = s.split('\n\n')
	ranges = sorted((int(rng.split('-')[0]), int(rng.split('-')[1]) + 1)
			for rng in ranges.splitlines())
	available = [int(n) for n in available.splitlines()]
	result1 = sum(any(a <= n < b for a, b in ranges) for n in available)
	result2 = 0
	a, b = 0, 0
	for aa, bb in ranges:
		if aa > b:
			result2 += b - a
			a, b = aa, bb
		elif bb > b:
			b = bb
	result2 += b - a
	return result1, result2


def day6(s):
	lines = s.splitlines()
	nums = [[int(a) for a in line.split()]
			for line in lines[:-1]]
	nums2 = [[]]
	for n, _ in enumerate(s.splitlines()[0]):
		if all(line[n] == ' ' for line in lines[:-1]):
			nums2.append([])
		else:
			nums2[-1].append(int(''.join(line[n] for line in lines[:-1])))
	ops = lines[-1].split()
	result1 = result2 = 0
	for n, op in enumerate(ops):
		if op == '+':
			result1 += sum(a[n] for a in nums)
			result2 += sum(nums2[n])
		elif op == '*':
			result1 += math.prod(a[n] for a in nums)
			result2 += math.prod(nums2[n])
		else:
			raise NotImplementedError(op)
	return result1, result2


def day7(s):
	def numsplitters(y, x):
		if y < len(grid):
			if grid[y][x] in '.S' and (y, x) not in seen:
				seen.add((y, x))
				return numsplitters(y + 1, x)
			elif grid[y][x] == '^':
				return 1 + numsplitters(y, x - 1) + numsplitters(y, x + 1)
		return 0

	@cache
	def numpaths(y, x):
		if y < len(grid):
			if grid[y][x] in '.S':
				return numpaths(y + 1, x)
			elif grid[y][x] == '^':
				return numpaths(y, x - 1) + numpaths(y, x + 1)
		return 1

	grid = s.splitlines()
	seen = set()
	result1 = numsplitters(0, grid[0].index('S'))
	result2 = numpaths(0, grid[0].index('S'))
	return result1, result2


def day8(s):
	boxes = [tuple(int(a) for a in line.split(',')) for line in s.splitlines()]
	numconnections = 10 if len(boxes) == 20 else 1000
	dists = sorted(itertools.combinations(range(len(boxes)), 2),
			key=lambda x: math.dist(boxes[x[0]], boxes[x[1]]))
	connections = [[n] for n, _coord in enumerate(boxes)]
	for n, (coord1, coord2) in enumerate(dists, 1):
		if connections[coord1] is not connections[coord2]:
			connections[coord1].extend(connections[coord2])
			for a in connections[coord2]:
				connections[a] = connections[coord1]
		if n == numconnections:
			circuits = {id(a): a for a in connections}.values()
			result1 = math.prod(sorted(len(a) for a in circuits)[-3:])
		if len(connections[coord1]) == len(boxes):
			result2 = boxes[coord1][0] * boxes[coord2][0]
			break
	return result1, result2


def day9(s):
	def pnpoly(tiles, x, y):
		# https://wrfranklin.org/Research/Short_Notes/pnpoly.html
		c = 0
		for (x1, y1), (x2, y2) in itertools.pairwise(tiles + [tiles[0]]):
			if ((y1 > y) != (y2 > y)
					and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
				c = not c
		return c

	def spanfill(tiles, grid, repl, verbose=False):
		coords = {}
		for y, x in sorted(grid):
			if y not in coords:
				coords[y] = []
			coords[y].append(x)
		ymax, xmax = max(grid)
		result = [[] for _ in range(ymax + 2)]
		for y, xs in coords.items():
			rngs = list(ranges(xs))
			for n, (rng1, rng2) in enumerate(zip(rngs, rngs[1:])):
				# if n & 1 == 0:
				if pnpoly(tiles, rng2[0] - 1, y):
					if result[y] and result[y][-1][1] >= rng1[0]:
						result[y][-1] = (result[y][-1][0], rng2[-1])
					else:
						result[y].append((rng1[0], rng2[-1]))
				else:
					if not result[y] or result[y][-1][1] < rng1[0]:
						result[y].append((rng1[0], rng1[-1]))
			if not result[y] or result[y][-1][-1] < rngs[-1][-1]:
				result[y].append((rngs[-1][0], rngs[-1][-1]))
			if verbose:
				print(y, xs, result[y])
			continue

			prev = -2
			start = -2
			count = 0
			for x in xs:
				if start == -2:
					start = x
				if prev + 1 != x:
					count += 1
					# if count and count & 1 == 0:
					if pnpoly(tiles, x - 1, y):
						result[y].append((start, x))
						start = -2
					prev = x
			print(y, xs, result[y])
		return result

	def ranges(s):
		"""Partition s into a lists corresponding to contiguous ranges.

		>>> list(ranges( (0, 1, 3, 4, 6) ))
		[[0, 1], [3, 4], [6]]"""
		rng = []
		for a in s:
			if not rng or a == rng[-1] + 1:
				rng.append(a)
			else:
				yield rng
				rng = [a]
		if rng:
			yield rng

	def checkxranges(y, x1, x2):
		x1, x2 = min(x1, x2), max(x1, x2)
		return any(xx1 <= x1 <= x2 <= xx2 for xx1, xx2 in xranges[y])

	def checkyranges(y, x1, x2):
		x1, x2 = min(x1, x2), max(x1, x2)
		return any(xx1 <= x1 <= x2 <= xx2 for xx1, xx2 in yranges[y])

	def dump():
		if len(tiles) > 100:
			return
		for y in range(ymax):
			print(y, ''.join(grid.get((y, x), '.') for x in range(xmax)))
		print()

	tiles = [tuple(map(int, line.split(','))) for line in s.splitlines()]
	result1 = max((abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
			for (x1, y1), (x2, y2) in itertools.combinations(tiles, 2))
	xmax, ymax = max(x for x, y in tiles) + 2, max(y for x, y in tiles) + 2
	grid = {}
	for x, y in tiles:
		grid[y, x] = '#'
	dump()
	for (x1, y1), (x2, y2) in itertools.pairwise(tiles + [tiles[0]]):
		if x1 == x2:
			for y in range(min(y1, y2) + 1, max(y1, y2)):
				grid[y, x1] = 'X'
		elif y1 == y2:
			for x in range(min(x1, x2) + 1, max(x1, x2)):
				grid[y1, x] = 'X'
		else:
			raise ValueError
	dump()
	print('fill')
	xranges = spanfill(tiles, grid, 'X', verbose=len(tiles) < 100)
	yranges = spanfill(
			[(y, x) for x, y in tiles],
			{(x, y): a for (y, x), a in grid.items()}, 'X')

	def dump(corners=()):
		if len(tiles) > 100:
			return
		for y in range(ymax):
			print(str(y).rjust(2), ''.join('O' if (y, x) in corners
					else grid.get((y, x),
						'x' if checkxranges(y, x, x) else '.')
					for x in range(xmax)), xranges[y])
		for x in range(xmax):
			print(str(x).rjust(2), yranges[x])
		print()

	dump()
	print('max')
	for (x1, y1), (x2, y2) in sorted(
			itertools.combinations(tiles, 2),
			key=lambda x: (abs(x[1][0] - x[0][0]) + 1) * (abs(x[1][1] - x[0][1]) + 1),
			reverse=True):
		# print(f'{(abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)} = {abs(x2 - x1) + 1} * {abs(y2 - y1) + 1}; {x1=} {y1=} {x2=} {y2=}', end=' ')
		if (checkxranges(y1, x1, x2)
				and checkxranges(y2, x1, x2)
				and checkyranges(x1, y1, y2)
				and checkyranges(x2, y1, y2)):
			result2 = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
			# print('yes', result2)
			break
		# print('no', checkxranges(y1, x1, x2),
		# 		checkxranges(y2, x1, x2),
		# 		checkyranges(x1, y1, y2),
		# 		checkyranges(x2, y1, y2))
	dump(corners=((y1, x1), (y2, x2)))
	print(f'{result2} = {abs(x2 - x1) + 1} * {abs(y2 - y1) + 1}; {x1=} {y1=} {x2=} {y2=}')
	return result1, result2


def test_day9_1():
	assert day9(open('i9test').read()) == (50, 24)
def test_day9_2():
	assert day9(open('i9test2').read()) == (35, 15)
def test_day9_3():
	assert day9(open('i9test3').read()) == (180, 30)
def test_day9_4():
	assert day9(open('i9test4').read()) == (90, 40)
def test_day9_5():
	assert day9(open('i9test5').read()) == (99, 35)
def test_day9_6():
	assert day9(open('i9test6').read()) == (256, 66)


def day10(s):
	import numpy
	import scipy
	result1 = result2 = 0
	for line in s.splitlines():
		indicators, rest = line.split(' ', 1)
		buttons, joltage = rest.rsplit(' ', 1)
		indicators = indicators.strip('[]').replace('.', '0').replace('#', '1')
		indicators = int(indicators[::-1], 2)
		buttons1 = [sum(1 << int(n) for n in a.strip('()').split(','))
				for a in buttons.split()]
		buttons2 = [[int(n) for n in a.strip('()').split(',')]
				for a in buttons.split()]
		joltage = [int(a) for a in joltage.strip('{}').split(',')]

		result = 9999
		for n in range(1, len(buttons1)):
			for seq in itertools.combinations(buttons1, n):
				state = 0
				for a in seq:
					state = (state & ~a) | (state ^ a)
				if state == indicators:
					result = min(result, len(seq))
			if result < 9999:
				break
		result1 += result

		buttons3 = [[n in button for button in buttons2]
				for n, _ in enumerate(joltage)]
		res = scipy.optimize.linprog(
				numpy.ones(len(buttons2)),
				A_eq=buttons3,
				b_eq=joltage,
				integrality=1)
		steps = round(res.fun)
		result2 += steps
	return result1, result2


def day11(s):
	@cache
	def paths(a, b):
		return sum(1 if loc == b else paths(loc, b)
				for loc in conn.get(a, ()))

	conn = {a: b.split() for a, b in
			(line.split(':') for line in s.splitlines())}
	result1 = paths('you', 'out')
	result2 = (paths('svr', 'dac') * paths('dac', 'fft') * paths('fft', 'out')
			+ paths('svr', 'fft') * paths('fft', 'dac') * paths('dac', 'out'))
	return result1, result2


def day12(s):
	*presents, regions = s.split('\n\n')
	presents = [present.splitlines()[1:] for present in presents]
	result1 = 0
	for region in regions.splitlines():
		area, shapes = region.split(':')
		width, heigth = map(int, area.split('x'))
		shapes = [int(a) for a in shapes.split()]
		result1 += sum(n * len(present) * len(present[0])
				for n, present in zip(shapes, presents)) <= width * heigth
	return result1


if __name__ == '__main__':
	main(globals())
