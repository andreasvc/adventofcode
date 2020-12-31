"""Advent of Code 2020. http://adventofcode.com/2020 """
import os
import re
import sys
import operator
from functools import reduce
# from collections import Counter, defaultdict, deque
# import numpy as np
# from numba import njit


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


def day3(s, right, down):
	trees = pos = 0
	for line in s.splitlines()[::down]:
		if line[pos % len(line.strip())] == '#':
			trees += 1
		pos += right
	return trees


def day3a(s):
	return day3(s, 3, 1)


def day3b(s):
	return (day3(s, 1, 1)
			* day3(s, 3, 1)
			* day3(s, 5, 1)
			* day3(s, 7, 1)
			* day3(s, 1, 2))


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


def day5(line):
	"""
	>>> day5('FBFBBFFRLR')
	357
	>>> day5('BFFFBBFRRR')
	567
	>>> day5('FFFBBBFRRR')
	119
	>>> day5('BBFFBBFRLL')
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
	return max(day5(line) for line in s.splitlines())


def day5b(s):
	seats = {8 * row + col
			for row in range(0, 128)
			for col in range(0, 8)}
	seats -= set(day5(line) for line in s.splitlines())
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


def day8(lines):
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
	return day8(lines)[1]


def day8b(s):
	lines = s.splitlines()
	for n, line in enumerate(lines):
		terminates = False
		if line.startswith(('jmp', 'nop')):
			other = 'jmp' if line.startswith('nop') else 'nop'
			newprog = lines[:n] + [other + line[3:]] + lines[n + 1:]
			terminates, acc = day8(newprog)
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
	return ...


def day10b(s):
	return ...


def day11a(s):
	return ...


def day11b(s):
	return ...


def day12a(s):
	return ...


def day12b(s):
	return ...


def day13a(s):
	return ...


def day13b(s):
	return ...


def day14a(s):
	return ...


def day14b(s):
	return ...


def day15a(s):
	return ...


def day15b(s):
	return ...


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


def day19a(s):
	return ...


def day19b(s):
	return ...


def day20a(s):
	return ...


def day20b(s):
	return ...


def day21a(s):
	return ...


def day21b(s):
	return ...


def day22a(s):
	return ...


def day22b(s):
	return ...


def day23a(s):
	return ...


def day23b(s):
	return ...


def day24a(s):
	return ...


def day24b(s):
	return ...


def day25a(s):
	return ...


def day25b(s):
	return ...


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
