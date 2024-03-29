"""Advent of Code 2019. http://adventofcode.com/2019 """
import re
import sys
import itertools
import math
# import operator
# from functools import reduce
from heapq import heappop, heappush
from collections import defaultdict  # Counter
import numpy as np
# from numba import njit
sys.path.append('..')
from common import main


def day1a(s):
	nums = [int(a) for a in s.splitlines()]
	return sum(a // 3 - 2 for a in nums)


def day1b(s):
	nums = [int(a) for a in s.splitlines()]
	total = 0
	for a in nums:
		fuel = a // 3 - 2
		while fuel > 0:
			total += fuel
			fuel = fuel // 3 - 2
	return total


def _day2(nums, a, b):
	nums[1] = a
	nums[2] = b
	pc = 0
	while True:
		op = nums[pc]
		if op == 1:
			nums[nums[pc + 3]] = nums[nums[pc + 1]] + nums[nums[pc + 2]]
			pc += 4
		elif op == 2:
			nums[nums[pc + 3]] = nums[nums[pc + 1]] * nums[nums[pc + 2]]
			pc += 4
		elif op == 99:
			break
		else:
			return -1  # raise ValueError
	return nums[0]


def day2a(s):
	nums = [int(a) for a in s.split(',')]
	return _day2(nums, 12, 2)


def day2b(s):
	for a in range(100):
		for b in range(100):
			nums = [int(a) for a in s.split(',')]
			if _day2(nums, a, b) == 19690720:
				return 100 * a + b


def day3a(s):
	"""
	>>> day3a('R75,D30,R83,U83,L12,D49,R71,U7,L72\\n'
	...		'U62,R66,U55,R34,D71,R55,D58,R83')
	159
	>>> day3a('R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51\\n'
	...		'U98,R91,D20,R16,D67,R40,U7,R15,U6,R7')
	135
	"""
	grid = {}
	dirs = {'L': (-1, 0), 'R': (1, 0),
			'U': (0, -1), 'D': (0, 1)}
	crossings = []
	for n, line in enumerate(s.splitlines(), 1):
		x = y = 0
		for pos in line.split(','):
			xd, yd = dirs[pos[0]]
			dist = int(pos[1:])
			if xd:
				for xx in range(x, x + xd * dist + xd, xd):
					if n == 1:
						grid[xx, y] = 1
					elif grid.get((xx, y)) == 1 and abs(xx) + abs(y):
						crossings.append(abs(xx) + abs(y))
				x += xd * dist
			elif yd:
				for yy in range(y, y + yd * dist + yd, yd):
					if n == 1:
						grid[x, yy] = n
					elif grid.get((x, yy)) == 1 and abs(x) + abs(yy):
						crossings.append(abs(x) + abs(yy))
				y += yd * dist
	return min(crossings)


def day3b(s):
	"""
	>>> day3b('R8,U5,L5,D3\\n'
	...		'U7,R6,D4,L4')
	30
	>>> day3b('R75,D30,R83,U83,L12,D49,R71,U7,L72\\n'
	...		'U62,R66,U55,R34,D71,R55,D58,R83')
	610
	>>> day3b('R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51\\n'
	...		'U98,R91,D20,R16,D67,R40,U7,R15,U6,R7')
	410
	"""
	dirs = {'L': (-1, 0), 'R': (1, 0),
			'U': (0, -1), 'D': (0, 1)}
	grids = [{}, {}]
	crossings = []
	for n, line in enumerate(s.splitlines()):
		x = y = steps = 0
		for pos in line.split(','):
			xd, yd = dirs[pos[0]]
			dist = int(pos[1:])
			if xd:
				for xx in range(x, x + xd * dist + xd, xd):
					grids[n][xx, y] = steps
					steps += 1
					if n == 1 and (xx, y) in grids[0] and abs(xx) + abs(y):
						crossings.append(grids[0][xx, y] + grids[1][xx, y])
				x += xd * dist
				steps -= 1
			elif yd:
				for yy in range(y, y + yd * dist + yd, yd):
					grids[n][x, yy] = steps
					steps += 1
					if n == 1 and (x, yy) in grids[0] and abs(x) + abs(yy):
						crossings.append(grids[0][x, yy] + grids[1][x, yy])
				y += yd * dist
				steps -= 1
	return min(crossings)


def day4a(s):
	a, b = [int(a) for a in s.split('-')]
	cnt = 0
	for n in range(a, b + 1):
		nn = str(n)
		if len(set(nn)) == 6:
			continue
		if all(d1 <= d2 for d1, d2 in zip(nn, nn[1:])):
			cnt += 1
	return cnt


def day4b(s):
	a, b = [int(a) for a in s.split('-')]
	digits = [(2 * str(d), 3 * str(d)) for d in range(10)]
	cnt = 0
	for n in range(a, b + 1):
		nn = str(n)
		if all(dd not in nn or ddd in nn for dd, ddd in digits):
			continue
		if all(d1 <= d2 for d1, d2 in zip(nn, nn[1:])):
			cnt += 1
	return cnt


def day5a(s):
	program = [int(a) for a in s.split(',')]
	return interpreter(program, [1])[0][-1]


def day5b(s):
	program = [int(a) for a in s.split(',')]
	return interpreter(program, [5])[0][-1]


def day6a(s):
	def visit(a):
		if a not in graph:
			return 0
		return sum(visit(b) + 1 for b in graph[a])

	sys.setrecursionlimit(100000)
	graph = defaultdict(set)
	for line in s.splitlines():
		a, b = line.split(')')
		graph[b].add(a)
	return sum(visit(a) for a in graph)


def day6b(s):
	def travel(a, b, visited=()):
		if a == b:
			return [0]
		elif a not in graph:
			return []
		return [d + 1
				for c in graph[a]
					if c not in visited
					for d in travel(c, b, visited + (a, ))]

	sys.setrecursionlimit(100000)
	graph = defaultdict(set)
	for line in s.splitlines():
		a, b = line.split(')')
		graph[b].add(a)
		graph[a].add(b)
	return min(travel('YOU', 'SAN')) - 2


def day7a(s):
	"""
	>>> day7a('3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0')
	43210
	>>> day7a('3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,'
	...		'23,99,0,0')
	54321
	>>> day7a('3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,'
	...		'33,1,33,31,31,1,32,31,31,4,31,99,0,0,0')
	65210
	"""
	program = parseprog(s)
	result = []
	for perm in itertools.permutations(range(5)):
		inp = [0]
		for n in perm:
			inp, _, _ = interpreter(program.copy(), [n] + inp[:1])
			# print(n, inp)
		# print(perm, inp)
		result.append(inp[0])
	return max(result)


def day7b(s):
	program = parseprog(s)
	result = []
	for perm in itertools.permutations(range(5, 10)):
		inp = [0]
		programs = [program.copy() for _ in range(5)]
		pcs = [0] * 5
		for n, phasesetting in enumerate(perm):
			inp, pcs[n], _ = interpreter(
					programs[n], [phasesetting] + inp,
					incremental=True, pc=pcs[n])
			if n == 4 and pcs[n] != -1:
				lastout = inp
		while pcs[-1] != -1:
			for n in range(5):
				inp, pcs[n], _ = interpreter(
						programs[n], inp, incremental=True, pc=pcs[n])
				if n == 4 and pcs[n] != -1:
					lastout = inp
		result.append(lastout[0])
	return max(result)


def day8a(s, width=25, height=6):
	layers = [s[n:n + width * height]
			for n in range(0, len(s), width * height)]
	cnt, layer = min((layer.count('0'), layer)
			for layer in layers)
	return layer.count('1') * layer.count('2')


def day8b(s, width=25, height=6):
	layers = [s[n:n + width * height]
			for n in range(0, len(s), width * height)]
	result = [2] * (width * height)
	for layer in layers:
		for n, char in enumerate(layer):
			if char in '01' and result[n] == 2:
				result[n] = int(char)
	return '\n'.join(
			''.join('#' if a else ' '
				for a in result[row * width:(row + 1) * width])
			for row in range(6))


def day9a(s):
	program = parseprog(s)
	return interpreter(program, [1], incremental=False)[0][-1]


def day9b(s):
	program = parseprog(s)
	return interpreter(program, [2], incremental=False)[0][-1]


def day10a(s):
	def lineofsight(x1, y1, x2, y2):
		if x1 == x2 and y1 == y2:
			return False
		xd, yd = x2 - x1, y2 - y1
		if xd == 0:
			if y1 > y2:
				y1, y2 = y2, y1
			return not grid[y1 + 1:y2, x1].any()
		elif yd == 0:
			if x1 > x2:
				x1, x2 = x2, x1
			return not grid[y1, x1 + 1:x2].any()
		div = math.gcd(xd, yd)
		stepx, stepy = xd // div, yd // div
		for x3, y3 in zip(
				range(x1 + stepx, x2, stepx),
				range(y1 + stepy, y2, stepy)):
			if grid[y3, x3]:
				return False
		return True

	def numasteroids(x, y):
		return sum(lineofsight(x, y, xx, yy)
				for yy, xx in zip(*grid.nonzero()))

	grid = np.array([[char == '#' for char in line]
			for line in s.splitlines()], dtype=int)
	return max((numasteroids(x, y), x, y) for y, x in zip(*grid.nonzero()))


def day10b(s):
	grid = np.array([[char == '#' for char in line]
			for line in s.splitlines()], dtype=int)
	_, x, y = day10a(s)  # origin
	coords = sorted(
			(math.atan2(yy - y, xx - x),  # theta
			-math.hypot(xx - x , yy - y),  # r
			xx, yy)
			for yy, xx in zip(*grid.nonzero()))
	# fixme: linear search
	for n, (theta, r, xx, yy) in enumerate(coords):
		if xx >= x:
			break
	coords = coords[n:] + coords[:n]
	coords = [list(group) for _, group
			in itertools.groupby(coords, lambda x: x[0])]
	_, _, xx, yy = next(itertools.islice(
			(group.pop() for group in itertools.cycle(coords) if group),
			199, None))
	return xx * 100 + yy


def day11a(s):
	program = parseprog(s)
	grid = defaultdict(int)
	pc = rb = direction = 0
	x = y = 2
	xd, yd = [0, 1, 0, -1], [-1, 0, 1, 0]
	while True:
		inp = [grid[x, y]]
		color, pc, rb = interpreter(
				program, inp, pc=pc, incremental=True, rb=rb)
		if pc == -1:
			break
		[turn], pc, rb = interpreter(
				program, inp, pc=pc, incremental=True, rb=rb)
		grid[x, y] = color[0]
		direction = (direction + (-1 if turn else 1)) % 4
		x, y = x + xd[direction], y + yd[direction]
	return len(grid)


def day11b(s):
	program = parseprog(s)
	grid = defaultdict(int)
	pc = rb = direction = 0
	x = y = 0
	grid[x, y] = 1
	xd, yd = [0, 1, 0, -1], [-1, 0, 1, 0]
	while True:
		inp = [grid[x, y]]
		color, pc, rb = interpreter(
				program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		[turn], pc, rb = interpreter(
				program, inp, incremental=True, pc=pc, rb=rb)
		grid[x, y] = color[0]
		direction = (direction + (-1 if turn else 1)) % 4
		x, y = x + xd[direction], y + yd[direction]
	xs, ys = [x for x, _ in grid], [y for _, y in grid]
	return '\n'.join(
			''.join('#' if grid[x, y] else ' '
				# why mirrored?
				for x in range(max(xs), min(xs) - 1, -1))
				# for x in range(min(xs), max(xs) + 1))
				for y in range(min(ys), max(ys) + 1))


def day12a(s, steps=1000):
	pos = np.array(list(map(int, re.findall(r'-?\d+', s))),
			dtype=int).reshape((-1, 3))
	vel = np.zeros_like(pos)
	for step in range(steps):
		vel += np.sign(pos - pos[:, np.newaxis]).sum(axis=1)
		pos += vel
	return np.abs(pos).sum(axis=1) @ np.abs(vel).sum(axis=1)


def day12b(s):
	pos = np.array(list(map(int, re.findall(r'-?\d+', s))),
			dtype=int).reshape((-1, 3))
	vel = np.zeros_like(pos)
	repeats = np.zeros(3, dtype=int)
	step = 0
	while (repeats == 0).any():
		vel += np.sign(pos - pos[:, np.newaxis]).sum(axis=1)
		pos += vel
		step += 1
		repeats[(repeats == 0) & (vel == 0).all(axis=0)] = 2 * step
	return np.lcm.reduce(repeats)


def day13a(s):
	program = parseprog(s)
	grid = {}
	pc = rb = 0
	while True:
		inp = []
		x, pc, rb = interpreter(program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		y, pc, rb = interpreter(program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		tileid, pc, rb = interpreter(
				program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		if tileid[0] == 2:
			grid[x[0], y[0]] = tileid[0]
	return len(grid)


def day13b(s):
	program = parseprog(s)
	program[0] = 2
	grid = defaultdict(int)
	pc = rb = score = paddlex= ballx = joystick = blocks = 0
	while True:
		inp = [joystick]
		x, pc, rb = interpreter(program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		y, pc, rb = interpreter(program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		tileid, pc, rb = interpreter(
				program, inp, incremental=True, pc=pc, rb=rb)
		if pc == -1:
			break
		if x[0] == -1 and y[0] == 0:
			score = tileid[0]
		else:
			if tileid[0] == 4:
				ballx = x[0]
			elif tileid[0] == 3:
				paddlex = x[0]
			elif tileid[0] == 2:
				blocks += 1
			elif tileid[0] == 0 and grid[x[0], y[0]] == 2:
				blocks -= 1
			grid[x[0], y[0]] = tileid[0]
		joystick = 0
		if paddlex < ballx:
			joystick = 1
		elif paddlex > ballx:
			joystick = -1
	return score


def day14a(s):
	reactions = {}
	for line in s.splitlines():
		inp, out = line.split(' => ')
		n, out = out.split()
		inp = [(int(a.split()[0]), a.split()[1]) for a in inp.split(', ')]
		assert out not in reactions
		reactions[out] = (int(n), inp)
	agenda = {'FUEL': 1}
	rest = {}
	ore = 0
	while agenda:
		chem = next(iter(agenda))
		n = agenda.pop(chem)
		if n == 0:
			continue
		m, inp = reactions[chem]
		if m < n:
			mult = n // m + ((n % m) != 0)
		else:
			mult = 1
		rest.setdefault(chem, 0)
		rest[chem] += mult * m - n
		for x, ch1 in inp:
			if ch1 == 'ORE':
				ore += x * mult
			else:
				amount = max(0, x * mult - rest.get(ch1, 0))
				agenda.setdefault(ch1, 0)
				agenda[ch1] += amount
				rest[ch1] = max(0, rest.get(ch1, 0) - (x * mult))
	return ore


def day14b(s):
	def run(agenda, rest, ore):
		while agenda:
			chem = next(iter(agenda))
			n = agenda.pop(chem)
			if n == 0:
				continue
			m, inp = reactions[chem]
			if m < n:
				mult = n // m + ((n % m) != 0)
			else:
				mult = 1
			rest.setdefault(chem, 0)
			rest[chem] += mult * m - n
			for x, ch1 in inp:
				if ch1 == 'ORE':
					ore += x * mult
				else:
					agenda.setdefault(ch1, 0)
					amount = max(0, x * mult - rest.get(ch1, 0))
					agenda[ch1] += amount
					rest[ch1] = max(0, rest.get(ch1, 0) - (x * mult))
		return ore

	reactions = {}
	for line in s.splitlines():
		inp, out = line.split(' => ')
		n, out = out.split()
		inp = [(int(a.split()[0]), a.split()[1]) for a in inp.split(', ')]
		assert out not in reactions
		reactions[out] = (int(n), inp)
	rest = {}
	oreperfuel = day14a(s)
	maxore = 1000000000000
	fuel = maxore // oreperfuel
	agenda = {'FUEL': fuel}
	ore = run(agenda, rest, 0)
	while ore < maxore:
		inc = (maxore - ore) // oreperfuel
		if inc == 0:
			break
		agenda = {'FUEL': inc}
		ore = run(agenda, rest, ore)
		fuel += inc
	return fuel


def _day15(s):
	import random
	random.seed(0)
	program = parseprog(s)
	walls = set()
	nonwalls = set()
	pc = rb = x = y = 0
	n = 1
	goalx = goaly = 0
	for _ in range(500_000):
		# if (y - 1, x) not in walls | nonwalls:
		# 	n = 1
		# elif (y, x + 1) not in walls | nonwalls:
		# 	n = 3
		# elif (y + 1, x) not in walls | nonwalls:
		# 	n = 2
		# elif (y, x - 1) not in walls | nonwalls:
		# 	n = 4
		# else:
		# 	n = random.choice([1, 2, 3, 4])
		n = random.choice([1, 2, 3, 4])
		# n = n % 4 + 1
		# print(n, len(walls), len(nonwalls))
		out, pc, rb = interpreter(program, [n], True, pc, rb)
		# change = False
		if out[0] == 0:
			xx, yy = x, y
			if n == 1:
				yy -= 1
			elif n == 2:
				yy += 1
			elif n == 3:
				xx += 1
			elif n == 4:
				xx -= 1
			walls.add((xx, yy))
		elif out[0] == 1:
			if n == 1:
				y -= 1
			elif n == 2:
				y += 1
			elif n == 3:
				x += 1
			elif n == 4:
				x -= 1
			nonwalls.add((x, y))
		elif out[0] == 2:
			if n == 1:
				y -= 1
			elif n == 2:
				y += 1
			elif n == 3:
				x += 1
			elif n == 4:
				x -= 1
			nonwalls.add((x, y))
			goalx, goaly = x, y
		else:
			raise ValueError
		# if change:
		# 	if n == 1:
		# 		n = 4  # north -> east
		# 	elif n == 2:
		# 		n = 3  # south -> west
		# 	elif n == 3:
		# 		n = 1  # west -> north
		# 	else:
		# 		n = 2  # east -> south
	# FIXME: need to explore ALL positions between (0, 0) and goal?
	return goalx, goaly, walls, nonwalls


def day15a(s):
	x, y, walls, nonwalls = _day15(s)
	agenda = [(abs(x) + abs(y), 0, x, y)]
	explored = set()
	while agenda:
		_, steps, x, y = heappop(agenda)
		if (x, y) in explored:
			continue
		explored.add((x, y))
		if (x, y) == (0, 0):
			return steps
		if (x + 1, y) not in walls and (x + 1, y) in nonwalls:
			heappush(agenda, (abs(x) + abs(y) + steps + 1, steps + 1, x + 1, y))
		if (x - 1, y) not in walls and (x - 1, y) in nonwalls:
			heappush(agenda, (abs(x) + abs(y) + steps + 1,steps + 1, x - 1, y))
		if (x, y + 1) not in walls and (x, y + 1) in nonwalls:
			heappush(agenda, (abs(x) + abs(y) + steps + 1,steps + 1, x, y + 1))
		if (x, y - 1) not in walls and (x, y - 1) in nonwalls:
			heappush(agenda, (abs(x) + abs(y) + steps + 1,steps + 1, x, y - 1))
	return steps


def day15b(s):
	x, y, walls, nonwalls = _day15(s)
	hasoxy = {(x, y)}
	new = [(x, y)]
	mins = 0
	# print(x, y)
	# minx = min(x for x, y in walls | nonwalls)
	# maxx = max(x for x, y in walls | nonwalls)
	# miny = min(y for x, y in walls | nonwalls)
	# maxy = max(y for x, y in walls | nonwalls)
	# for y in range(miny, maxy + 1):
	# 	print(''.join('#' if (x, y) in walls
	# 			else '.' if (x, y) in nonwalls
	# 			else ' '
	# 			for x in range(minx, maxx + 1)))
	#
	while hasoxy != nonwalls:
		newnew = set()
		for x, y in new:
			newnew.update({(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)})
		newnew &= nonwalls
		newnew -= hasoxy
		hasoxy.update(newnew)
		new = newnew
		mins += 1
	return mins


def day16a(s, phases=100):
	nums = np.array([int(a) for a in s], dtype=int)
	patterns = []
	for m, _ in enumerate(nums, 1):
		patterns.append(np.tile(np.array([0, 1, 0, -1], dtype=np.int8
				).repeat(m), len(nums) // (4 * m) + 1)[1:len(nums) + 1])
	patterns = np.array(patterns)
	for n in range(1, phases + 1):
		nums = np.abs(nums @ patterns.T) % 10
	return ''.join(str(a) for a in nums[:8])


def day16b(s, phases=100):
	nums = np.tile(np.array([int(a) for a in s], dtype=int), 10000)
	size = len(nums)
	offset = int(''.join(str(a) for a in nums[:7]))
	nums = nums[offset:]
	nums = np.append(nums, [0])
	for _ in range(phases):
		for m in range(size - offset - 1, -1, -1):
			nums[m] += nums[m + 1]
		nums %= 10
	return ''.join(str(a) for a in nums[:8])


def day17a(s):
	program = parseprog(s)
	out, pc, rb = interpreter(program, [])
	out = ''.join(chr(a) for a in out).strip()
	grid = np.array([list(a) for a in out.splitlines()])
	result = 0
	for y, x in zip(*((grid == '#')
			& (np.roll(grid, (0, -1), axis=1) == '#')
			& (np.roll(grid, (0, 1), axis=1) == '#')
			& (np.roll(grid, (-1, 0), axis=0) == '#')
			& (np.roll(grid, (1, 0), axis=0) == '#')).nonzero()):
		grid[y, x] = 'O'
		result += y * x
	# print('\n'.join(
	# 	''.join(grid[y, x] for x in range(grid.shape[1]))
	# 	for y in range(grid.shape[0])))
	return result


def _day17b_path(s):
	import tty
	import sys
	import termios
	program = parseprog(s)
	out, pc, rb = interpreter(program, [])
	out = ''.join(chr(a) for a in out).strip()
	grid = np.array([list(a) for a in out.splitlines()])
	program[0] = 2
	(y, ), (x, ) = (grid == '^').nonzero()
	dir = 0
	robot = '^>v<'
	dirkeys = ',eoa'
	steps = []
	orig_settings = termios.tcgetattr(sys.stdin)
	try:
		tty.setcbreak(sys.stdin)  # https://stackoverflow.com/a/34497639
		while True:
			print('\n'.join(
				''.join(grid[y, x] for x in range(grid.shape[1]))
				for y in range(grid.shape[0])))
			print('move (,=up o=down a=left e=right):')
			key = sys.stdin.read(1)[0]
			origx, origy, origdir = x, y, dir
			if key in dirkeys:
				dir = dirkeys.index(key)
				if key == ',':  # up
					y -= 1
				elif key == 'e':  # right
					x += 1
				elif key == 'o':  # down
					y += 1
				elif key == 'a':  # left
					x -= 1
			elif key == '.' or key == chr(27):  # stop
				print('stopping')
				break
			else:  # wrong key
				print('unrecognized command:', repr(key))
				continue
			if grid[y, x] in '#%':
				grid[origy, origx] = '%'
				grid[y, x] = robot[dir]
				# steps.append(robot[dir])
				if dir == origdir:
					steps[-1] += 1
				else:
					if dir - origdir > 0:
						steps.append('R' * (dir - origdir))
						if steps[-1] == 'RRR':
							steps[-1] = 'L'
					else:
						steps.append('L' * (origdir - dir))
						if steps[-1] == 'LLL':
							steps[-1] = 'R'
					steps.append(1)
				print(','.join(str(a) for a in steps))
			else:
				print('no # at', y, x, grid[y, x], grid[origy, origx])
				x, y, dir = origx, origy, origdir
	finally:
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)


def day17b(s):
	program = parseprog(s)
	program[0] = 2
	# full path, manually obtained with _day17b_path():
	path = ('R,8,L,12,R,8,R,8,L,12,R,8,L,10,L,10,R,8,L,12,L,12,L,10,R,10,'
			'L,10,L,10,R,8,L,12,L,12,L,10,R,10,L,10,L,10,R,8,R,8,L,12,R,8,'
			'L,12,L,12,L,10,R,10,R,8,L,12,R,8')
	# manually compressed:
	funcs = {'A': 'R,8,L,12,R,8',
			'B': 'L,10,L,10,R,8',
			'C': 'L,12,L,12,L,10,R,10'}
	for a, b in funcs.items():
		path = path.replace(b, a)
	inp = [ord(a) for a in path] + [10]
	for a in 'ABC':
		inp += [ord(b) for b in funcs[a]] + [10]
	inp += [ord('n'), 10]
	out, pc, rb = interpreter(program, inp)
	result = out[-1]
	out = ''.join(chr(a) for a in out[:-1]).strip()
	# print(out)
	return result


def day19a(s):
	program = parseprog(s)
	result = 0
	for y in range(50):
		for x in range(50):
			out, pc, rb = interpreter(program.copy(), [x, y])
			result += out[0] == 1
	return result


def day19b(s, gridsize=2000, ship=100, step=50):
	def explore(y):
		if not grid[y, :].any():
			start = False
			minx = 0
			if y and grid[y - 1, :].any():
				minx = grid[y - 1, :].nonzero()[0][0]
			for x in range(minx, gridsize):
				out, pc, rb = interpreter(program.copy(), [x, y])
				grid[y, x] = out[0]
				if out[0] and not start:
					start = True
				elif not out[0] and start:
					break

	program = parseprog(s)
	grid = np.zeros((gridsize, gridsize), dtype=bool)
	y = 0
	while True:
		explore(y)
		explore(y + ship - 1)
		x = grid[y + ship - 1, :].nonzero()[0][0]
		if grid[y, x + ship - 1] and grid[y + ship - 1, x]:
			y -= 2 * step
			break
		y += step
	while True:
		explore(y)
		explore(y + ship - 1)
		x = grid[y + ship - 1, :].nonzero()[0][0]
		if grid[y, x + ship - 1] and grid[y + ship - 1, x]:
			return x * 10000 + y
		y += 1


def parseprog(s):
	return defaultdict(int, enumerate(int(a) for a in s.split(',')))


def interpreter(nums, inp, incremental=False, pc=0, rb=0):
	"""day 9 version. returns for every output if incremental==True."""
	def read(par):
		if mode[-par] == '0':  # position
			if nums[pc + par] < 0:
				raise ValueError
			return nums[nums[pc + par]]
		elif mode[-par] == '1':  # immediate
			if pc + par < 0:
				raise ValueError
			return nums[pc + par]
		elif mode[-par] == '2':  # relative base
			if nums[pc + par] + rb < 0:
				raise ValueError
			return nums[nums[pc + par] + rb]
		raise ValueError

	def write(par, val):
		if mode[-par] == '0':  # position
			if nums[pc + par] < 0:
				raise ValueError
			nums[nums[pc + par]] = val
		elif mode[-par] == '2':  # relative base
			if nums[pc + par] + rb < 0:
				raise ValueError
			nums[nums[pc + par] + rb] = val
		else:
			raise ValueError

	if pc < 0 or pc not in nums:
		raise ValueError
	outputs = []
	while True:
		op = nums[pc]
		mode = '%03d' % (op // 100)
		op = op % 100
		if op == 99:  # halt
			break
		elif op == 3:  # input
			write(1, inp.pop(0))
			pc += 2
		elif op == 4:  # output
			a = read(1)
			outputs.append(a)
			pc += 2
			if incremental:
				return outputs, pc, rb
		elif op in [1, 2, 5, 6, 7, 8]:  # add/mult
			a, b = read(1), read(2)
			if op == 1:  # add
				write(3, a + b)
			elif op == 2:  # mult
				write(3, a * b)
			elif op == 7:  # less than
				write(3, int(a < b))
			elif op == 8:  # equals
				write(3, int(a == b))
			# jump if true, jump if zero
			if (op == 5 and a != 0) or (op == 6 and a == 0):
				pc = b
			else:
				pc += 3 if 5 <= op <= 6 else 4
		elif op == 9:  # adjust rb
			rb += read(1)
			pc += 2
		else:
			raise ValueError(op)
	return outputs, -1, rb  # -1=halt

if __name__ == '__main__':
	main(globals())
