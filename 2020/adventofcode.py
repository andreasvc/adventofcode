"""Advent of Code 2020. http://adventofcode.com/2020 """
import re
import sys
import operator
import itertools
from functools import reduce
from collections import Counter, defaultdict
import numpy as np
from numba import njit
sys.path.append('..')
from common import main


def day1a(s):
	nums = [int(a) for a in s.splitlines()]
	for a in nums:
		for b in nums:
			if a + b == 2020:
				return a * b


def day1b(s):
	nums = [int(a) for a in s.splitlines()]
	for a in nums:
		for b in nums:
			for c in nums:
				if a + b + c == 2020:
					return a * b * c


def day2a(s):
	valid = 0
	for line in s.splitlines():
		policy, password = line.split(': ', 1)
		rng, char = policy.split(' ')
		a, b = rng.split('-')
		a, b = int(a), int(b)
		if a <= password.count(char) <= b:
			valid += 1
	return valid


def day2b(s):
	valid = 0
	for line in s.splitlines():
		policy, password = line.split(': ', 1)
		rng, char = policy.split(' ')
		a, b = rng.split('-')
		a, b = int(a), int(b)
		if (password[a - 1] == char) + (password[b - 1] == char) == 1:
			valid += 1
	return valid


def _day3(s, right, down):
	trees = pos = 0
	for line in s.splitlines()[::down]:
		if line[pos % len(line.strip())] == '#':
			trees += 1
		pos += right
	return trees


def day3a(s):
	return _day3(s, 3, 1)


def day3b(s):
	return (_day3(s, 1, 1)
			* _day3(s, 3, 1)
			* _day3(s, 5, 1)
			* _day3(s, 7, 1)
			* _day3(s, 1, 2))


def day4a(s):
	valid = 0
	for passport in s.split('\n\n'):
		fields = passport.split()
		keys = {a.split(':')[0] for a in fields}
		if set('byr iyr eyr hgt hcl ecl pid'.split()) <= keys:
			valid += 1
	return valid


def day4b(s):
	valid = 0
	for passport in s.split('\n\n'):
		passport = dict(a.split(':') for a in passport.split())
		if (set('byr iyr eyr hgt hcl ecl pid'.split()) <= passport.keys()
				and 1920 <= int(passport['byr']) <= 2002
				and 2010 <= int(passport['iyr']) <= 2020
				and 2020 <= int(passport['eyr']) <= 2030
				and ((passport['hgt'].endswith('cm')
						and 150 <= int(passport['hgt'][:-2]) <= 193)
					or (passport['hgt'].endswith('in')
						and 59 <= int(passport['hgt'][:-2]) <= 76))
				and re.match(r'^#[a-f0-9]{6}$', passport['hcl'])
				and passport['ecl'] in 'amb blu brn gry grn hzl oth'.split()
				and re.match(r'^[0-9]{9}$', passport['pid'])
				):
			valid += 1
	return valid


def _day5(line):
	"""
	>>> _day5('FBFBBFFRLR')
	357
	>>> _day5('BFFFBBFRRR')
	567
	>>> _day5('FFFBBBFRRR')
	119
	>>> _day5('BBFFBBFRLL')
	820"""
	minrow, maxrow = 0, 128
	for a in line[:7]:
		if a == 'F':
			maxrow -= (maxrow - minrow) // 2
		elif a == 'B':
			minrow += (maxrow - minrow) // 2
		else:
			raise ValueError
	mincol, maxcol = 0, 8
	for a in line[7:].strip():
		if a == 'R':
			mincol += (maxcol - mincol) // 2
		elif a == 'L':
			maxcol -= (maxcol - mincol) // 2
		else:
			raise ValueError
	return 8 * minrow + mincol


def day5a(s):
	return max(_day5(line) for line in s.splitlines())


def day5b(s):
	seats = {8 * row + col
			for row in range(0, 128)
			for col in range(0, 8)}
	seats -= set(_day5(line) for line in s.splitlines())
	for a in seats:
		if a - 1 not in seats and a + 1 not in seats:
			return a


def day6a(s):
	result = 0
	for group in s.split('\n\n'):
		result += len(set(group.replace('\n', '')))
	return result


def day6b(s):
	result = 0
	for group in s.split('\n\n'):
		lines = [set(line) for line in group.splitlines()]
		result += len(lines[0].intersection(*lines))
	return result


def day7a(s):
	rules = {}  # key: bag x, values: list of bags which may contain x
	for line in s.splitlines():
		lhs, rhs = line.strip().split(' contain ')
		lhs = ' '.join(lhs.split()[:-1])
		for a in rhs.split(', '):
			color = ' '.join(a.split()[1:-1])
			if color not in rules:
				rules[color] = set()
			rules[color].add(lhs)
	# start from shiny gold bag, compute transitive closure
	queue = list(rules['shiny gold'])
	visited = {'shiny gold'}
	while queue:
		color = queue.pop()
		visited.add(color)
		queue.extend(a for a in rules.get(color, ()) if a not in visited)
	return len(visited - {'shiny gold'})


def day7b(s):
	rules = {}  # key: bag x, values: list of bags in x
	for line in s.splitlines():
		lhs, rhs = line.strip().split(' contain ')
		lhs = ' '.join(lhs.split()[:-1])
		for a in rhs.split(', '):
			if a.startswith('no '):
				continue
			num = int(a.split()[0])
			color = ' '.join(a.split()[1:-1])
			if lhs not in rules:
				rules[lhs] = set()
			rules[lhs].add((num, color))
	queue = list(rules['shiny gold'])
	visited = {'shiny gold'}
	result = 0
	while queue:
		num, color = queue.pop()
		visited.add(color)
		result += num
		queue.extend((num * n, a) for n, a in rules.get(color, ()))
	return result


def _day8(lines):
	acc = pos = 0
	executed = set()
	while pos not in executed and pos < len(lines):
		executed.add(pos)
		line = lines[pos]
		if line.startswith('acc'):
			acc += int(line.split()[1])
			pos += 1
		elif line.startswith('jmp'):
			pos += int(line.split()[1])
		elif line.startswith('nop'):
			pos += 1
	return pos >= len(lines), acc


def day8a(s):
	lines = s.splitlines()
	return _day8(lines)[1]


def day8b(s):
	lines = s.splitlines()
	for n, line in enumerate(lines):
		terminates = False
		if line.startswith(('jmp', 'nop')):
			other = 'jmp' if line.startswith('nop') else 'nop'
			newprog = lines[:n] + [other + line[3:]] + lines[n + 1:]
			terminates, acc = _day8(newprog)
			if terminates:
				return acc


def day9a(s):
	numbers = [int(a) for a in s.splitlines()]
	for n, num in enumerate(numbers):
		if n < 25:
			continue
		if not any(a + b == num
				for a in numbers[n - 25:n]
					for b in numbers[n - 25:n]
					if a != b):
			return num


def day9b(s):
	numbers = [int(a) for a in s.splitlines()]
	goal = day9a(s)
	for length in range(2, len(numbers) + 1):
		for n, num in enumerate(numbers):
			if sum(numbers[n:n + length]) == goal:
				return min(numbers[n:n + length]) + max(numbers[n:n + length])


def day10a(s):
	numbers = [0] + [int(a) for a in s.split()]
	numbers.append(max(numbers) + 3)
	numbers.sort()
	cnt = Counter(b - a for a, b in zip(numbers, numbers[1:]))
	return cnt[1] * cnt[3]


def day10b(s):
	# https://old.reddit.com/r/adventofcode/comments/kacdbl/2020_day_10c_part_2_no_clue_how_to_begin/gf9lzhd/
	numbers = sorted(int(a) for a in s.split())
	numbers = [0] + numbers + [max(numbers) + 3]
	result = [0] * len(numbers)
	result[0] = 1
	for n, a in enumerate(numbers):
		for m in range(n + 1, min(n + 4, len(numbers))):
			if numbers[m] - a <= 3:
				result[m] += result[n]
	return result[-1]


def day11a(s):
	def nb(grid, x, y):
		return sum(grid[yy][xx] == 2
				for yy in range(max(0, y - 1), min(len(grid), y + 2))
					for xx in range(max(0, x - 1), min(len(grid[0]), x + 2)))

	newgrid = [[1 if char == 'L' else 0 for char in line]
			for line in s.splitlines()]
	grid = None
	while newgrid != grid:
		grid = newgrid
		newgrid = [row.copy() for row in grid]
		for y, row in enumerate(grid):
			for x, elem in enumerate(row):
				if grid[y][x] == 0:
					continue
				nbs = nb(grid, x, y)
				if grid[y][x] == 1 and nbs == 0:
					newgrid[y][x] = 2
				elif grid[y][x] == 2 and nbs > 4:
					newgrid[y][x] = 1
	return sum(1 for row in grid for elem in row if elem == 2)


def day11b(s):
	def nb(grid, x, y):
		nbs = 0
		for dx in [-1, 0, 1]:
			for dy in [-1, 0, 1]:
				if dx == dy == 0:
					continue
				xx = x + dx
				yy = y + dy
				while (0 <= yy < len(grid) and 0 <= xx < len(grid[0])
						and grid[yy][xx] == 0):
					xx += dx
					yy += dy
				nbs += (0 <= yy < len(grid) and 0 <= xx < len(grid[0])
						and grid[yy][xx] == 2)
		return nbs

	newgrid = [[1 if char == 'L' else 0 for char in line]
			for line in s.splitlines()]
	grid = None
	while newgrid != grid:
		grid = newgrid
		newgrid = [row.copy() for row in grid]
		for y, row in enumerate(grid):
			for x, elem in enumerate(row):
				if grid[y][x] == 0:
					continue
				nbs = nb(grid, x, y)
				if grid[y][x] == 1 and nbs == 0:
					newgrid[y][x] = 2
				elif grid[y][x] == 2 and nbs > 4:
					newgrid[y][x] = 1
	return sum(elem == 2 for row in grid for elem in row)


def day12a(s):
	x = y = 0
	dirx, diry = 1, 0
	for cmd in s.splitlines():
		op, amount = cmd[0], int(cmd[1:])
		if op == 'N':
			y -= amount
		elif op == 'S':
			y += amount
		elif op == 'E':
			x += amount
		elif op == 'W':
			x -= amount
		elif op == 'L':
			if amount == 180:
				dirx, diry = -dirx, -diry
			elif amount == 90:
				dirx, diry = diry, -dirx
			elif amount == 270:
				dirx, diry = -diry, dirx
		elif op == 'R':
			if amount == 180:
				dirx, diry = -dirx, -diry
			elif amount == 90:
				dirx, diry = -diry, dirx
			elif amount == 270:
				dirx, diry = diry, -dirx
		elif op == 'F':
			x += dirx * amount
			y += diry * amount
	return abs(x) + abs(y)


def day12b(s):
	x = y = 0
	wpx, wpy = 10, -1
	for cmd in s.splitlines():
		op, amount = cmd[0], int(cmd[1:])
		if op == 'N':
			wpy -= amount
		elif op == 'S':
			wpy += amount
		elif op == 'E':
			wpx += amount
		elif op == 'W':
			wpx -= amount
		elif op == 'L':
			if amount == 180:
				wpx, wpy = -wpx, -wpy
			elif amount == 90:
				wpx, wpy = wpy, -wpx
			elif amount == 270:
				wpx, wpy = -wpy, wpx
		elif op == 'R':
			if amount == 180:
				wpx, wpy = -wpx, -wpy
			elif amount == 90:
				wpx, wpy = -wpy, wpx
			elif amount == 270:
				wpx, wpy = wpy, -wpx
		elif op == 'F':
			x += wpx * amount
			y += wpy * amount
	return abs(x) + abs(y)


def day13a(s):
	timestamp, buses = s.splitlines()
	timestamp = int(timestamp)
	buses = [int(a) for a in buses.split(',') if a != 'x']
	earliestbus = min((bus for bus in buses),
			key=lambda bus: bus - timestamp % bus)
	minutes = earliestbus - timestamp % earliestbus
	return earliestbus * minutes


def day13b(s):
	fields = s.splitlines()[1].split(',')
	ind = [n for n, a in enumerate(fields) if a != 'x']
	buses = [int(a) for a in fields if a != 'x']
	# https://old.reddit.com/r/adventofcode/comments/kc4njx/2020_day_13_solutions/gfnwnf3/
	t = period = 0
	for n in range(1, len(buses) + 1):
		while not all((t + n) % a == 0 for n, a in zip(ind[:n], buses[:n])):
			t += period
		period = reduce(operator.mul, buses[:n], 1)
	return t


@njit
def _day13b():
	# Brute force. Took 1:41:02.37
	# The input:
	# ind = [0,  35,  41, 49, 54, 58, 70,  72, 91]
	# bus = [41, 37, 541, 23, 13, 17, 29, 983, 19]
	period = 983
	t = period - 72
	while True:
		if (
				(t + 41) % 541 == 0
				and t % 41 == 0
				and (t + 35) % 37 == 0
				and (t + 70) % 29 == 0
				and (t + 49) % 23 == 0
				and (t + 91) % 19 == 0
				and (t + 58) % 17 == 0
				and (t + 54) % 13 == 0
				# and (t + 72) % 983 == 0
				):
			return t
		t += period


def day14a(s):
	mem = {}
	onemask, zeromask = 0, (1 << 36) - 1
	for line in s.splitlines():
		lhs, val = line.split(' = ')
		if lhs == 'mask':
			onemask = sum(1 << n for n, a in enumerate(val[::-1]) if a == '1')
			zeromask = ((1 << 36) - 1) ^ sum(
					1 << n for n, a in enumerate(val[::-1]) if a == '0')
		elif lhs.startswith('mem['):
			mem[int(lhs[4:-1])] = int(val) & zeromask | onemask
	return sum(mem.values())


def day14b(s):
	mem = {}
	onemask = 0
	floating = []
	floatmask = ((1 << 36) - 1)
	for line in s.splitlines():
		lhs, val = line.split(' = ')
		if lhs == 'mask':
			onemask = sum(1 << n for n, a in enumerate(val[::-1]) if a == '1')
			floating = [n for n, a in enumerate(val[::-1]) if a == 'X']
			floatmask = ((1 << 36) - 1) ^ sum(1 << n for n in floating)
		elif lhs.startswith('mem['):
			loc = int(lhs[4:-1]) & floatmask | onemask
			for comb in itertools.chain.from_iterable(
					itertools.combinations(floating, length)
					for length in range(len(floating) + 1)):
				mem[loc | sum(1 << n for n in comb)] = int(val)
	return sum(mem.values())


def day15a(s):
	numbers = [int(a) for a in s.split(',')]
	while len(numbers) < 2020:
		if numbers[-1] not in numbers[:-1]:
			numbers.append(0)
		else:
			numbers.append(numbers[:-1][::-1].index(numbers[-1]) + 1)
	return numbers[2019]


def day15b(s):
	numbers = np.array([int(a) for a in s.split(',')], dtype=np.int32)
	return _day15b(numbers)


@njit
def _day15b(numbers, goal=30_000_000):
	spoken = np.zeros(goal + 1, dtype=np.int32)
	for n, num in enumerate(numbers[:-1], 1):
		spoken[num] = n
	num = numbers[-1]
	nn = len(numbers)
	while nn < goal:
		prev = num
		num = 0 if spoken[num] == 0 else nn - spoken[num]
		spoken[prev] = nn
		nn += 1
	return num


def day16a(s):
	rules, your, nearby = s.split('\n\n')
	rules = {n
			for line in rules.splitlines()
				for rng in line.split(': ')[1].split(' or ')
					for n in range(
						int(rng[:rng.index('-')]),
						int(rng[rng.index('-') + 1:]))}
	return sum(int(num)
			for line in nearby.splitlines()[1:]
				for num in line.split(',')
					if int(num) not in rules)


def day16b(s):
	rules, your, nearby = s.split('\n\n')
	valid = {n
			for line in rules.splitlines()
				for rng in line.split(': ')[1].split(' or ')
					for n in range(
						int(rng[:rng.index('-')]),
						int(rng[rng.index('-') + 1:]) + 1)}
	rules = {line.split(':')[0]: {n
				for rng in line.split(': ')[1].split(' or ')
					for n in range(
						int(rng[:rng.index('-')]),
						int(rng[rng.index('-') + 1:]) + 1)}
			for line in rules.splitlines()}
	your = [int(num) for num in your.splitlines()[1].split(',')]
	nearby = [[int(num) for num in line.split(',')]
			for line in nearby.splitlines()[1:]
			if all(int(num) in valid
				for num in line.split(','))]
	result = {}
	assigned = set()
	while len(result) < len(rules):
		for n, num in enumerate(your):
			if n in assigned:
				continue
			values = {num} | {ticket[n] for ticket in nearby}
			candidates = sum(1 for key, numbers in rules.items()
					if key not in result and values <= numbers)
			if candidates == 1:
				for key, numbers in rules.items():
					if key not in result and values <= numbers:
						result[key] = num
						assigned.add(n)
						break
			elif candidates == 0:
				raise ValueError
	return reduce(operator.mul,
			(result[key] for key in result if key.startswith('departure')), 1)


def day17a(s):
	grid = {}
	for y, line in enumerate(s.splitlines()):
		for x, char in enumerate(line.strip()):
			if char == '#':
				grid[x, y, 0] = 1
	for cycle in range(6):
		newgrid = {}
		minx = min(x for x, y, z in grid) - 1
		maxx = max(x for x, y, z in grid) + 2
		miny = min(y for x, y, z in grid) - 1
		maxy = max(y for x, y, z in grid) + 2
		minz = min(z for x, y, z in grid) - 1
		maxz = max(z for x, y, z in grid) + 2
		for x in range(minx, maxx):
			for y in range(miny, maxy):
				for z in range(minz, maxz):
					nb = 0
					for xd in range(x - 1, x + 2):
						for yd in range(y - 1, y + 2):
							for zd in range(z - 1, z + 2):
								if xd == x and yd == y and zd == z:
									continue
								nb += grid.get((xd, yd, zd), 0)
					if (grid.get((x, y, z), 0) == 1 and 2 <= nb <= 3) or (
							grid.get((x, y, z), 0) == 0 and nb == 3):
						newgrid[x, y, z] = 1
		grid = newgrid
	return sum(grid.values())


def day17b(s):
	def getnb(grid, x, y, z):
		nb = 0
		for xd in range(x - 1, x + 2):
			for yd in range(y - 1, y + 2):
				for zd in range(z - 1, z + 2):
					for wd in range(w - 1, w + 2):
						nb += grid.get((xd, yd, zd, wd), 0)
						if nb > 4:
							return nb
		return nb

	grid = {}
	for y, line in enumerate(s.splitlines()):
		for x, char in enumerate(line.strip()):
			if char == '#':
				grid[x, y, 0, 0] = 1
	for cycle in range(6):
		newgrid = {}
		minx = min(x for x, y, z, w in grid) - 1
		maxx = max(x for x, y, z, w in grid) + 2
		miny = min(y for x, y, z, w in grid) - 1
		maxy = max(y for x, y, z, w in grid) + 2
		minz = min(z for x, y, z, w in grid) - 1
		maxz = max(z for x, y, z, w in grid) + 2
		minw = min(w for x, y, z, w in grid) - 1
		maxw = max(w for x, y, z, w in grid) + 2
		active = 0
		for x in range(minx, maxx):
			for y in range(miny, maxy):
				for z in range(minz, maxz):
					for w in range(minw, maxw):
						nb = getnb(grid, x, y, z)
						if (3 <= nb <= 4 and grid.get((x, y, z, w), 0) == 1) or (
								nb == 3 and grid.get((x, y, z, w), 0) == 0):
							newgrid[x, y, z, w] = 1
							active += 1
		grid = newgrid
	return active


def day18a(s):
	def myeval(line):
		if line.strip().isdigit():
			val, rest = line, None
		elif line[-1] == ')':
			par = 1
			pos = len(line) - 2
			while par:
				if line[pos] == ')':
					par += 1
				elif line[pos] == '(':
					par -= 1
				pos -= 1
				if pos < 0 and par:
					raise ValueError('unbalanced parentheses: %r' % line)
			val = myeval(line[pos + 2:len(line) - 1])
			rest = line[:pos + 1].rstrip()
		else:
			if ' ' not in line:
				raise ValueError('unbalanced parentheses: %r' % line)
			rest, val = line.rsplit(' ', 1)
		if not rest:
			result = int(val)
		elif rest[-1] == '+':
			result = myeval(rest[:-1].rstrip()) + int(val)
		elif rest[-1] == '*':
			result = myeval(rest[:-1].rstrip()) * int(val)
		else:
			raise ValueError(repr(line))
		return result

	return sum(myeval(line) for line in s.splitlines())


def day18b(s):
	# https://en.wikipedia.org/wiki/Operator-precedence_parser#Alternative_methods
	def preproc(line):
		res = '((('
		tokens = list(line.replace(' ', ''))
		for n, tok in enumerate(tokens):
			if tok == '(':
				res += '((('
			elif tok == ')':
				res += ')))'
			elif tok == '+':
				res += ') + ('
			elif tok == '*':
				res += ')) * (('
			else:
				res += tok
		return res + ')))'

	return day18a('\n'.join(preproc(line) for line in s.splitlines()))


def _slow_day19a(s):
	import nltk
	rules, received = s.split('\n\n')
	rules = sorted(rules.splitlines(),
				key=lambda x: not x.startswith('0: '))
	grammar = nltk.CFG.fromstring(
			line.replace(':', ' ->', 1)
			for line in rules)
	parser = nltk.ChartParser(grammar)
	result = 0
	for n, line in enumerate(received.splitlines()):
		res = parser.parse(list(line))
		try:
			_ = next(iter(res))
			result += 1
		except StopIteration:
			pass
	return result


def day19a(s):
	def conv(rules):
		for line in rules.splitlines():
			lhs, rhs = line.split(':', 1)
			for alt in rhs.split(' | '):
				if '"' in alt:
					yield (((lhs, 'Epsilon'), (alt.strip().strip('"'), )), 1)
				else:
					alt = tuple(alt.split())
					yield (((lhs, ) + alt, (tuple(range(len(alt))), )), 1)

	# https://github.com/andreasvc/disco-dop
	from discodop import containers, pcfg
	rules, received = s.split('\n\n')
	grammar = containers.Grammar(list(conv(rules)), start='0')
	return sum(bool(pcfg.parse(list(line), grammar)[0])
			for line in received.splitlines())


def day19b(s):
	# manually binarized
	return day19a(
			s.replace('8: 42\n', '8: 42 | 42 8\n'
			).replace('11: 42 31\n', '11: 42 31 | 42 11x\n11x: 11 31\n'))


def day20a(s):
	tiles = {
			int(tile[:tile.index(':')].split()[1]):
			np.array([[1 if char == '#' else 0 for char in line]
				for line in tile[tile.index(':') + 2:].splitlines()])
			for tile in s.split('\n\n')}
	mapping = defaultdict(list)
	for tileid, tile in tiles.items():
		sides = [tile[:, 0], tile[:, -1], tile[0, :], tile[-1, :]]
		sides = [min(list(s), list(s)[::-1]) for s in sides]
		for s in sides:
			mapping[tuple(s)].append(tileid)
	# the 4 corners are the tiles with two unique sides
	cnt = Counter(b[0] for b in mapping.values() if len(b) == 1)
	return reduce(operator.mul, [a for a, b in cnt.items() if b == 2], 1)


def day20b(s):
	def parse(img):
		return np.array([[1 if char == '#' else 0 for char in line]
				for line in img.splitlines()], dtype=bool)

	def vec(seq):  # convert a boolean vector to an integer
		return min(
				sum(1 << n for n, a in enumerate(seq) if a),
				sum(1 << n for n, a in enumerate(seq[::-1]) if a))

	def find(monster, im):
		found = 0
		for y in range(0, im.shape[0] - monster.shape[0]):
			for x in range(0, im.shape[1] - monster.shape[1]):
				part = im[y:y + monster.shape[0], x:x + monster.shape[1]]
				if ((part & monster) == monster).all():
					im[y:y + monster.shape[0], x:x + monster.shape[1]] &= ~(
							monster)
					found += 1
		return found

	tiles = {int(tile[:tile.index(':')].split()[1]):
				parse(tile[tile.index(':') + 2:])
			for tile in s.split('\n\n')}
	mapping = defaultdict(set)
	for tileid, tile in tiles.items():
		for side in [tile[:, 0], tile[:, -1], tile[0, :], tile[-1, :]]:
			mapping[vec(side)].add(tileid)
	# the 4 corners are the tiles with two unique sides
	cnt = Counter(next(iter(b)) for b in mapping.values() if len(b) == 1)
	# pick an arbitrary corner as top left
	corner = [a for a, b in cnt.items() if b == 2].pop()
	tile = tiles[corner]
	# find the right rotation: the right and bottom sides must have neighbors
	for _ in range(4):
		if (len(mapping[vec(tile[-1, :])]) == 2
				and len(mapping[vec(tile[:, -1])]) == 2):
			break
		tile = np.rot90(tile)
	# put tile at top left of result
	ylen = tile.shape[0] - 2
	xlen = tile.shape[1] - 2
	xtiles = ytiles = int(len(tiles) ** 0.5)
	im = np.zeros((ytiles * ylen, xtiles * xlen), dtype=bool)
	empty = np.ones((ytiles * ylen, xtiles * xlen), dtype=bool)
	im[:ylen, :xlen] = tile[1:-1, 1:-1]
	empty[:ylen, :xlen] = np.zeros((ylen, xlen), dtype=bool)
	xcur, ycur = xlen, 0
	prev = tile
	used = {corner}
	# approach: spiral from outside in; always add new tiles to the right of
	# previous tile; when a tile without right neighbor is reached
	# (corner) or it already has a neighbor, change direction by rotating.
	while True:
		# does the previous tile have a neighbor that has not been placed?
		while mapping[vec(prev[:, -1])] - used:
			a, b = mapping[vec(prev[:, -1])]
			match = a if b in used else b
			tile = tiles[match]
			# find the right rotation
			for _ in range(4):
				if (prev[:, -1] == tile[:, 0]).all():
					break
				elif (prev[:, -1] == tile[:, 0][::-1]).all():
					tile = np.flipud(tile)
					break
				tile = np.rot90(tile)
			# add to image
			im[ycur:ycur + ylen, xcur:xcur + xlen] = tile[1:-1, 1:-1]
			empty[ycur:ycur + ylen, xcur:xcur + xlen] = np.zeros((ylen, xlen),
					dtype=bool)
			xcur += xlen
			prev = tile
			used.add(match)
		if len(used) == len(tiles):
			break
		# switch direction
		im = np.rot90(im)
		empty = np.rot90(empty)
		prev = np.rot90(prev)
		yy, xx = empty.nonzero()
		ycur, xcur = yy[0], xx[0]

	monster = parse(
			'                  # \n'
			'#    ##    ##    ###\n'
			' #  #  #  #  #  #   \n')
	for _ in range(2):
		for _ in range(4):
			if find(monster, im):
				return im.sum()
			im = np.rot90(im)
		im = np.fliplr(im)


def _day21(s):
	ingredients = [set(line.split('(contains ')[0].split())
			for line in s.splitlines()]
	allergens = [set(line.split('(contains ')[1].rstrip(')').split(', '))
			for line in s.splitlines()]
	candidates = {aller: set(a)
			for a, b in zip(ingredients, allergens)
			for aller in b}
	for a, b in zip(ingredients, allergens):
		for aller in b:
			candidates[aller] &= a
	return ingredients, allergens, candidates


def day21a(s):
	ingredients, allergens, candidates = _day21(s)
	return len([ingr for a, b in zip(ingredients, allergens)
			for ingr in a
			if not any(ingr in x for x in candidates.values())])


def day21b(s):
	ingredients, allergens, candidates = _day21(s)
	while any(len(b) > 1 for b in candidates.values()):
		for a, b in candidates.items():
			if len(b) == 1:
				for aa, bb in candidates.items():
					if aa != a:
						candidates[aa] -= b
	return ','.join(next(iter(candidates[a])) for a in sorted(candidates))


def day22a(s):
	deck1, deck2 = s.split('\n\n')
	deck1 = [int(a) for a in deck1.splitlines()[1:]]
	deck2 = [int(a) for a in deck2.splitlines()[1:]]
	while deck1 and deck2:
		a, b = deck1.pop(0), deck2.pop(0)
		if a > b:
			deck1.extend(sorted([a, b], reverse=True))
		else:
			deck2.extend(sorted([a, b], reverse=True))
	return sum(n * a for n, a in enumerate((deck1 or deck2)[::-1], 1))


def day22b(s):
	def subgame(deck1, deck2):
		seen = set()
		while deck1 and deck2:
			if (tuple(deck1), tuple(deck2)) in seen:
				deck2 = []
				break
			seen.add((tuple(deck1), tuple(deck2)))
			a, b = deck1.pop(0), deck2.pop(0)
			if len(deck1) >= a and len(deck2) >= b:
				winner = subgame(deck1[:a].copy(), deck2[:b].copy())
			else:
				winner = a > b
			if winner:
				deck1.extend([a, b])
			else:
				deck2.extend([b, a])
		return winner

	deck1, deck2 = s.split('\n\n')
	deck1 = [int(a) for a in deck1.splitlines()[1:]]
	deck2 = [int(a) for a in deck2.splitlines()[1:]]
	_ = subgame(deck1, deck2)
	return sum(n * a for n, a in enumerate((deck1 or deck2)[::-1], 1))


def day23a(s):
	cups = [int(a) for a in list(s.strip())]
	mincup, maxcup = min(cups), max(cups)
	for move in range(100):
		pickup = cups[1:4]
		cups[1:4] = []
		dest = cups[0] - 1
		while dest in pickup or dest < mincup:
			dest -= 1
			if dest < mincup:
				dest = maxcup
		idx = cups.index(dest)
		cups[idx + 1:idx + 1] = pickup
		cups = cups[1:] + cups[:1]
	idx = cups.index(1)
	cups = cups[idx + 1:] + cups[:idx]
	return ''.join(str(a) for a in cups)


def day23b(s, maxcup=1_000_000):
	start = np.array([int(a) for a in s.strip()], dtype=np.int32)
	return _day23b(start)


@njit
def _day23b(start, maxcup=1_000_000):
	# cups[cup] == nextcup
	cups = np.zeros(maxcup + 1, dtype=np.int32)
	for a, b in zip(start, start[1:]):
		cups[a] = b
	cups[start[-1]] = 10
	for n in range(10, maxcup):
		cups[n] = n + 1
	cur = start[0]
	cups[maxcup] = cur
	for _ in range(10_000_000):
		a = cups[cur]
		b = cups[a]
		c = cups[b]
		dest = cur - 1
		while dest == a or dest == b or dest == c or dest < 1:
			dest -= 1
			if dest < 1:
				dest = maxcup
		cups[cur], cups[dest], cups[c] = cups[c], a, cups[dest]
		cur = cups[cur]
	return int(cups[1]) * int(cups[cups[1]])


def _day24(s):
	stepre = re.compile('(nw|ne|e|se|sw|w)')
	state = {}
	for line in s.splitlines():
		x = y = z = 0
		for step in stepre.findall(line):
			if step == 'e':
				x += 1
				y -= 1
			elif step == 'w':
				x -= 1
				y += 1
			elif step == 'ne':
				x += 1
				z -= 1
			elif step == 'nw':
				y += 1
				z -= 1
			elif step == 'se':
				y -= 1
				z += 1
			elif step == 'sw':
				x -= 1
				z += 1
		state[x, y, z] = not state.get((x, y, z), 0)
	return state


def day24a(s):
	state = _day24(s)
	return sum(state.values())


def day24b(s):
	def nbs(x, y, z):
		return [(x + 1, y - 1, z), (x - 1, y + 1, z),
				(x + 1, y, z - 1), (x, y + 1, z - 1),
				(x, y - 1, z + 1), (x - 1, y, z + 1)]

	state = _day24(s)  # 0=white(default), 1=black
	for day in range(100):
		newstate = {}
		for (x, y, z), val in state.items():
			blacknbs = sum(state.get(nb, 0) for nb in nbs(x, y, z))
			if val == 1:
				newstate[x, y, z] = (0
						if blacknbs == 0 or blacknbs > 2 else 1)
			elif blacknbs == 2:
				newstate[x, y, z] = 1
			for nb in nbs(x, y, z):
				if state.get(nb, 0) == 1:
					continue
				blacknbs = sum(state.get(nb1, 0) for nb1 in nbs(*nb))
				if blacknbs == 2:
					newstate[nb] = 1
		state = newstate
	return sum(state.values())


def day25a(s):
	def transform(subject):
		val = 1
		for n in itertools.count():
			val *= subject
			val %= 20201227
			yield n + 1, val

	a, b = s.splitlines()
	cardpub, doorpub = int(a), int(b)
	for n, val in transform(7):
		if val == cardpub:
			_n, secretkey = next(iter(itertools.islice(transform(doorpub),
					n - 1, None)))
			break
		elif val == doorpub:
			_n, secretkey = next(iter(itertools.islice(transform(cardpub),
					n - 1, None)))
			break
	return secretkey


def day25b(s):
	"""There is no 25b."""


if __name__ == '__main__':
	main(globals())
