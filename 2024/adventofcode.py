"""Advent of Code 2024. http://adventofcode.com/2024 """
import re
import sys
from heapq import heappop, heappush
from functools import cache, cmp_to_key
from math import log10
from itertools import product, combinations
from collections import Counter, deque
from array import array
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
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
	def myeval(nums, result, concat):
		if len(nums) == 1:
			return nums[0] == result
		num = nums[-1]
		if num < result and myeval(nums[:-1], result - num, concat):
			return True
		elif result % num == 0 and myeval(nums[:-1], result // num, concat):
			return True
		elif concat:
			ndigits = 10 ** int(log10(num) + 1)
			if result % ndigits == num and myeval(
					nums[:-1], result // ndigits, concat):
				return True
		return False


	result1 = result2 = 0
	for line in s.splitlines():
		outcome, nums = line.split(':')
		outcome, nums = int(outcome), [int(a) for a in nums.split()]
		result1 += outcome * myeval(nums, outcome, False)
		result2 += outcome * myeval(nums, outcome, True)
	return result1, result2


def day8(s):
	grid = s.splitlines()
	ymax, xmax = len(grid), len(grid[0])
	ants = {}
	for y, line in enumerate(grid):
		for x, char in enumerate(line):
			if char != '.':
				if char not in ants:
					ants[char] = []
				ants[char].append((y, x))
	result1, result2 = set(), set()
	for char in ants:
		for (y1, x1), (y2, x2) in combinations(ants[char], 2):
			for n in range(-100, 100):
				y3 = y1 - n * (y2 - y1)
				x3 = x1 - n * (x2 - x1)
				if 0 <= y3 < ymax and 0 <= x3 < xmax:
					if n in (-2, 1):
						result1.add((y3, x3))
					result2.add((y3, x3))
	return len(result1), len(result2)


def day9(s):
	data = []
	for (n, a), b in zip(enumerate(s[::2]), s[1::2]):
		data.extend([n for _ in range(int(a))])
		data.extend([-1 for _ in range(int(b))])
	data.extend([n + 1 for _ in range(int(s[-1]))])
	disk = array('i', data)
	start, end = 0, len(disk) - 1
	while True:
		for n in range(end, -1, -1):
			if disk[n] != -1:
				end = n - 1
				break
		for m, a in enumerate(disk[start:], start):
			if a == -1 and m < n:
				start = m + 1
				disk[n], disk[m] = disk[m], disk[n]
				break
		else:
			break
	result1 = sum(n * int(a) for n, a in enumerate(disk) if a != -1)

	disk = array('i', data)
	freelist = []
	n = 0
	while n < len(disk):
		if disk[n] == -1:
			nn = n
			while disk[nn] == -1:
				nn += 1
			freelist.append((n, nn))
			n = nn
		else:
			n += 1
	end = len(disk) - 1
	while True:
		for nn in range(end, -1, -1):
			if disk[nn] != -1:
				n = nn
				while disk[n] == disk[nn]:
					n -= 1
				end = n
				nn += 1
				n += 1
				blocks = nn - n
				break
		if n == 0:
			break
		for x, (m, mm) in enumerate(freelist):
			if m > n:
				break
			elif mm - m >= blocks:
				disk[n:nn], disk[m:m + blocks] = disk[m:m + blocks], disk[n:nn]
				if mm - m == blocks:
					freelist.pop(x)
				else:
					freelist[x] = (m + blocks, mm)
				break
	result2 = sum(n * int(a) for n, a in enumerate(disk) if a != -1)
	return result1, result2


def day10(s):
	grid = {(x, y): -1 if a == '.' else int(a)
			for y, line in enumerate(s.splitlines())
			for x, a in enumerate(line)}
	dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
	nines = {(x, y): set() for (x, y), a in grid.items() if a == 0}
	paths = {(x, y): set() for (x, y), a in grid.items() if a == 0}
	agenda = [(a, ) for a in nines]
	while agenda:
		path = agenda.pop()
		if len(path) == 10:
			nines[path[0]].add(path[-1])
			paths[path[0]].add(path)
			continue
		x, y = path[-1]
		for dx, dy in dirs:
			if grid.get((x + dx, y + dy)) == grid.get((x, y), -1) + 1:
				agenda.append(path + ((x + dx, y + dy), ))
	result1 = sum(len(a) for a in nines.values())
	result2 = sum(len(a) for a in paths.values())
	return result1, result2


def day11(s):
	nums = Counter([int(a) for a in s.split()])
	for n in range(75):
		newnums = Counter()
		for a, b in nums.items():
			if a == 0:
				newnums[1] += b
				continue
			digits = (int(log10(a)) + 1)
			if (digits & 1) == 0:
				newnums[a // 10 ** (digits // 2)] += b
				newnums[a % 10 ** (digits // 2)] += b
			else:
				newnums[a * 2024] += b
			nums = newnums
			if n == 24:
				result1 = sum(nums.values())
	return result1, sum(nums.values())


def day12(s):
	grid = {(x, y): char
			for y, line in enumerate(s.splitlines())
				for x, char in enumerate(line)}
	groups = DisjointSet(grid)
	for x, y in grid:
		for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
			if grid[x, y] == grid.get((x + dx, y + dy)):
				groups.merge((x, y), (x + dx, y + dy))
	regions = groups.subsets()
	areas = [len(a) for a in regions]
	perimeters = [
			(4 * len(region)) - sum(grid[x, y] == grid.get((x + dx, y + dy))
				for x, y in region
				for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)])
			for region in regions]
	sides = []
	for region in regions:
		regionsides = []
		for x, y in region:
			for dy in (-1, 1):
				side = set()
				for dx in (-1, 1):
					xx = x
					while (xx, y) in region and grid.get(
							(xx, y)) != grid.get((xx, y + dy)):
						side.add((xx, y + 0.5 * dy))
						xx += dx
				if side:
					regionsides.append(tuple(sorted(side)))
			for dx in (-1, 1):
				side = set()
				for dy in (-1, 1):
					yy = y
					while (x, yy) in region and grid.get(
							(x, yy)) != grid.get((x + dx, yy)):
						side.add((x + 0.5 * dx, yy))
						yy += dy
				if side:
					regionsides.append(tuple(sorted(side)))
		sides.append(len(set(regionsides)))
	result1 = sum(a * b for a, b in zip(areas, perimeters))
	result2 = sum(a * b for a, b in zip(areas, sides))
	return result1, result2


def day13(s):
	def solve(ax, ay, bx, by, gx, gy):
		a = np.array([[ax, bx], [ay, by]], dtype=int)
		b = np.array([gx, gy], dtype=int)
		a1, a2 = a.copy(), a.copy()
		a1[:, 0] = b
		a2[:, 1] = b
		deta = np.linalg.det(a)
		n = round(np.linalg.det(a1) / deta)
		m = round(np.linalg.det(a2) / deta)
		if (b - a @ np.array([n, m], dtype=int) == 0).all():
			return 3 * n + m
		return 0

	result1 = result2 = 0
	for machine in s.split('\n\n'):
		nums = re.findall(r'\d+', machine)
		ax, ay, bx, by, gx, gy = [int(a) for a in nums]
		result1 += solve(ax, ay, bx, by, gx, gy)
		gx += 10000000000000
		gy += 10000000000000
		result2 += solve(ax, ay, bx, by, gx, gy)
	return result1, result2


def day14(s, verbose=False):
	data = np.array([int(a) for a in re.findall(r'-?\d+', s)],
			dtype=int).reshape((-1, 4))
	pos = data[:, :2]
	vel = data[:, 2:]
	size = np.array([11, 7], dtype=int)
	if (pos > size).any():
		size = np.array([101, 103], dtype=int)
	pos += 100 * vel
	pos %= size
	mid = size // 2
	mid1 = size // 2 + size % 2
	ul = (pos < mid).all(axis=1)
	lr = (pos >= mid1).all(axis=1)
	ur = (pos[:, 0] >= mid1[0]) & (pos[:, 1] < mid[1])
	ll = (pos[:, 0] < mid[0]) & (pos[:, 1] >= mid1[1])
	result1 = ul.sum() * ur.sum() * ll.sum() * lr.sum()

	result2 = 0
	for n in range(101, 100000):
		pos += vel
		pos %= size
		if 2 * (pos[:, 1] < mid[1]).sum() < (pos[:, 1] >= mid1[1]).sum():
			cnt = Counter([(x, y) for x, y in pos])
			if any('1111111111111111111111111111111'
					in ''.join(str(cnt[x, y] or '.') for x in range(size[0]))
					for y in range(size[1])):
				if verbose:
					for y in range(size[1]):
						print(''.join(str(cnt[x, y] or '.') for x in range(size[0])))
				result2 = n
				break
	return result1, result2


def day15(s, verbose=False):
	def canmove(grid, x, y, dx, dy):
		if grid[x + dx, y + dy] == '#':
			return False
		elif grid[x + dx, y + dy] == '.':
			return True
		elif grid[x + dx, y + dy] == 'O':
			return canmove(grid, x + dx, y + dy, dx, dy)
		elif grid[x + dx, y + dy] in '[]':
			if dx == 0:
				if grid[x + dx, y + dy] in '[':
					return canmove(grid, x, y + dy, dx, dy
							) and canmove(grid, x + 1, y + dy, dx, dy)
				else:
					return canmove(grid, x, y + dy, dx, dy
							) and canmove(grid, x - 1, y + dy, dx, dy)
			else:
				return canmove(grid, x + dx, y + dy, dx, dy)

	def domove(grid, x, y, dx, dy):
		if grid[x + dx, y + dy] == 'O':
			domove(grid, x + dx, y + dy, dx, dy)
		elif grid[x + dx, y + dy] in '[]':
			if dx == 0:
				if grid[x + dx, y + dy] in '[':
					domove(grid, x + 1, y + dy, dx, dy)
				else:
					domove(grid, x - 1, y + dy, dx, dy)
			domove(grid, x + dx, y + dy, dx, dy)
		grid[x + dx, y + dy], grid[x, y] = grid[x, y], grid[x + dx, y + dy]

	def solve(grid):
		ymax, xmax = len(grid.splitlines()), len(grid.splitlines()[0])
		grid = {(x, y): a
				for y, line in enumerate(grid.splitlines())
				for x, a in enumerate(line)}
		x, y = next(iter((x, y) for (x, y), c in grid.items() if c == '@'))
		dxdy = {'^': (0, -1), 'v': (0, 1), '<': (-1, 0), '>': (1, 0)}
		for move in movements:
			dx, dy = dxdy[move]
			if canmove(grid, x, y, dx, dy):
				domove(grid, x, y, dx, dy)
				x, y = x + dx, y + dy
			if verbose:
				print('\nMove:', move)
				for yy in range(ymax):
					print(''.join(grid.get((xx, yy),' ') for xx in range(xmax)))
		return sum(x + 100 * y
				for (x, y), char in grid.items()
				if char in 'O[')

	grid, movements = s.split('\n\n')
	movements = movements.replace('\n', '')
	grid1 = (grid.replace('#', '##').replace('O', '[]')
				.replace('.', '..').replace('@', '@.'))
	return solve(grid), solve(grid1)


def day16(s):
	grid = s.splitlines()
	start = max((line.find('S'), y) for y, line in enumerate(grid))
	end = max((line.find('E'), y) for y, line in enumerate(grid))
	# agenda = [(end[0] + end[1], 0) + start + (1, 0, (start, ))]
	agenda = [(0, 0) + start + (1, 0, (start, ))]
	seen = {start + (1, 0): 0}  # end[0] + end[1]}
	bestcost = 99999999999
	onbestpath = set()
	while agenda:
		est, cost, x, y, dx, dy, path = heappop(agenda)
		if (x, y) == end:
			if cost > bestcost:
				return bestcost, len(onbestpath)
			if cost < bestcost:
				bestcost = cost
			onbestpath.update(path)
		if seen[x, y, dx, dy] < est:
			continue
		for ddx, ddy, dcost in [(dx, dy, 1), (dy, dx, 1001), (-dy, -dx, 1001)]:
			ny, nx, ncost = y + ddy, x + ddx, cost + dcost
			if grid[ny][nx] != '#':
				est = ncost  # + abs(end[0] - nx) + abs(end[1] - ny)
				if est <= seen.get((nx, ny, ddx, ddy), 99999999):
					heappush(agenda, (est, ncost, nx, ny, ddx, ddy,
							path + ((nx, ny), )))
					seen[nx, ny, ddx, ddy] = est


def day17(s):
	def run(A, B, C, prog):
		out = []
		ip = 0
		while ip < len(prog):
			op = prog[ip]
			operand = prog[ip + 1]
			opval = operand
			if operand == 4:
				opval = A
			elif operand == 5:
				opval = B
			elif operand == 6:
				opval = C
			if op == 0:
				A >>= opval  # A //= 2 ** opval
			elif op == 1:
				B ^= operand
			elif op == 2:
				B = opval & 0b111  # % 8
			elif op == 3:
				if A != 0:
					ip = operand
					continue
			elif op == 4:
				B ^= C
			elif op == 5:
				out.append(opval & 0b111)  # % 8)
			elif op == 6:
				B = A >> opval  # A // 2 ** opval
			elif op == 7:
				C = A >> opval  # A // 2 ** opval
			ip += 2
		return out

	def solve(prog, n, result):
		for m in range(0, 0b1000):
			out = run(result | m, 0, 0, prog)
			if n < len(out) and out[-(n + 1)] == prog[-(n + 1)]:
				if n == len(prog) - 1:
					return result | m
				newresult = solve(prog, n + 1, (result | m) << 3)
				if newresult is not None:
					return newresult

	regs, prog = s.split('\n\n')
	prog = [int(a) for a in re.findall(r'\d+', prog)]
	A, B, C = [int(a) for a in re.findall(r'\d+', regs)]
	result1 = ','.join(str(a) for a in run(A, B, C, prog))
	result2 = solve(prog, 0, 0)
	return result1, result2


def day18(s):
	incoming = [tuple(int(b) for b in a.split(',')) for a in s.splitlines()]
	xmax, ymax = 71, 71
	corrupted, rest = set(incoming[:1024]), incoming[1024:]
	if not rest:
		xmax, ymax = 7, 7
		corrupted, rest = set(incoming[:12]), incoming[12:]
	start = 0, 0
	end = xmax - 1, ymax - 1
	dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

	def find():
		agenda = [(0, ) + start + ((start, ), )]
		seen = {c: 0 for c in corrupted | {start}}
		while agenda:
			cost, x, y, path = heappop(agenda)
			if (x, y) == end:
				return cost, path
			if seen[x, y] < cost:
				continue
			for dx, dy in dirs:
				nx, ny, ncost = x + dx, y + dy, cost + 1
				if (0 <= nx < xmax and 0 <= ny < ymax
						and ncost < seen.get((nx, ny), 99999999)):
					heappush(agenda, (ncost, nx, ny, path + ((nx, ny), )))
					seen[nx, ny] = ncost
		return None, ()

	result1, path = find()
	for coord in rest:
		corrupted.add(coord)
		if coord in path:
			cost, path = find()
			if cost is None:
				return result1, coord


def day19(s):
	@cache
	def nummatch(design):
		if design == '':
			return 1
		return sum(nummatch(design[len(pat):]) for pat in patterns
				if design.startswith(pat))

	import re2 as re
	patterns, designs = s.split('\n\n')
	patterns = patterns.split(', ')
	designs = designs.splitlines()
	patternre = re.compile('^(%s)+$' % '|'.join(patterns))
	result1 = sum(patternre.match(design) is not None for design in designs)
	result2 = sum(nummatch(design) for design in designs)
	return result1, result2


def day20(s):
	def find(start, end):
		agenda = deque([start])
		seen = {start: 0}
		while agenda:
			x, y = agenda.popleft()
			if (x, y) == end:
				return seen
			for dx, dy in dirs:
				nx, ny, ncost = x + dx, y + dy, seen[x, y] + 1
				if 0 <= nx < xmax and 0 <= ny < ymax:
					if grid[ny][nx] != '#':
						if ncost < seen.get((nx, ny), 99999999):
							agenda.append((nx, ny))
							seen[nx, ny] = ncost

	grid = s.splitlines()
	xmax, ymax = len(grid[0]), len(grid)
	start = max((line.find('S'), y) for y, line in enumerate(grid))
	end = max((line.find('E'), y) for y, line in enumerate(grid))
	dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
	dist = find(start, end)
	result1 = result2 = 0
	for x1, y1 in dist:
		for x2 in range(x1 - 20, x1 + 21):
			yd = 20 - abs(x2 - x1)
			for y2 in range(y1 - yd, y1 + yd + 1):
				d = abs(x2 - x1) + abs(y2 - y1)
				if 2 <= d <= 20 and (x2, y2) in dist:
					diff = dist[x1, y1] - dist[x2, y2] - d
					if diff >= 100:
						if d == 2:
							result1 += 1
						result2 += 1
	return result1, result2


def day21(s):
	def find(code, pad):
		x, y = pad['A']
		bx, by = pad[' ']
		result = ''
		for c in code:
			nx, ny = pad[c]
			yxbad = ((x == bx and min(y, ny) <= by <= max(y, ny))
					or (ny == by and min(x, nx) <= bx <= max(x, nx)))
			xybad = ((y == by and min(x, nx) <= bx <= max(x, nx))
					or (nx == bx and min(y, ny) <= by <= max(y, ny)))
			if not yxbad and not xybad and y > ny:
				result += (abs(x - nx) * ('<' if x > nx else '>')
						+ abs(y - ny) * ('^' if y > ny else 'v') + 'A')
			elif not yxbad:
				result += (abs(y - ny) * ('^' if y > ny else 'v')
						+ abs(x - nx) * ('<' if x > nx else '>') + 'A')
			else:
				result += (abs(x - nx) * ('<' if x > nx else '>')
						+ abs(y - ny) * ('^' if y > ny else 'v') + 'A')
			x, y = nx, ny
		return result

	def find2(code, n):
		return sum(f(c0, c, n) for c0, c in zip('A' + code, code))

	@cache
	def f(c0, c, n):
		x, y = dpad[c0]
		nx, ny = dpad[c]
		yxbad = ((x == bx and min(y, ny) <= by <= max(y, ny))
				or (ny == by and min(x, nx) <= bx <= max(x, nx)))
		xybad = ((y == by and min(x, nx) <= bx <= max(x, nx))
				or (nx == bx and min(y, ny) <= by <= max(y, ny)))
		if not yxbad and not xybad and n:
			result1 = (abs(x - nx) * ('<' if x > nx else '>')
					+ abs(y - ny) * ('^' if y > ny else 'v') + 'A')
			result2 = (abs(y - ny) * ('^' if y > ny else 'v')
					+ abs(x - nx) * ('<' if x > nx else '>') + 'A')
			return min(
					sum(f(a, b, n - 1)
						for a, b in zip('A' + result1, result1)),
					sum(f(a, b, n - 1)
						for a, b in zip('A' + result2, result2)))
		elif not yxbad:
			result = (abs(y - ny) * ('^' if y > ny else 'v')
					+ abs(x - nx) * ('<' if x > nx else '>') + 'A')
		else:
			result = (abs(x - nx) * ('<' if x > nx else '>')
					+ abs(y - ny) * ('^' if y > ny else 'v') + 'A')
		if n == 0:
			return len(result)
		return sum(f(a, b, n - 1)
				for a, b in zip('A' + result, result))


	npad = '789\n456\n123\n 0A'
	dpad = ' ^A\n<v>'
	npad, dpad = [{c: (x, y) for y, line in enumerate(pad.splitlines())
			for x, c in enumerate(line)} for pad in [npad, dpad]]
	bx, by = dpad[' ']
	result1 = result2 = 0

	for code in s.splitlines():
		result1 += find2(find(code, npad), 1) * int(code[:-1])
		result2 += find2(find(code, npad), 24) * int(code[:-1])
	return result1, result2


def day22(s):
	result1 = 0
	price = Counter()
	for line in s.splitlines():
		num = prev = int(line)
		changes = []
		seen = set()
		for n in range(2000):
			num ^= num << 6  # * 64
			num &= 16777216 - 1  # num %= 16777216
			num ^= num >> 5  # // 32
			num &= 16777216 - 1  # num %= 16777216
			num ^= num << 11 # * 2048
			num &= 16777216 - 1  # num %= 16777216
			num10 = num % 10
			changes.append(num10 - prev)
			prev = num10
			if n >= 4:
				seq = tuple(changes[-4:])
				if seq not in seen:
					seen.add(seq)
					if num10:
						price[seq] += num10
		result1 += num
	result2 = price.most_common()[0][1]
	return result1, result2


def day23(s):
	conn = {}
	for line in s.splitlines():
		a, b = line.split('-')
		if a not in conn:
			conn[a] = set()
		conn[a].add(b)
		if b not in conn:
			conn[b] = set()
		conn[b].add(a)
	result1 = set()
	for a in conn:
		if a[0] == 't':
			for b, c in combinations(conn[a], 2):
				if b in conn[c]:
					result1.add(tuple(sorted((a, b, c))))

	def bronkerbosch(r, p, x):
		if not p and not x:
			yield r
		for v in list(p):
			yield from bronkerbosch(r | {v}, p & conn[v], x & conn[v])
			p.discard(v)
			x.add(v)

	maximumclique = max(bronkerbosch(set(), set(conn), set()), key=len)
	result2 = ','.join(sorted(maximumclique))
	return len(result1), result2


def day24(s):
	def run(wires, circuit):
		todo = [True] * len(circuit)
		change = True
		while any(todo) and change:
			change = False
			for n, (a, op, b, _, out) in enumerate(circuit):
				if not todo[n] or a not in wires or b not in wires:
					continue
				a, b = wires[a], wires[b]
				if op == 'AND':
					wires[out] = a and b
				elif op == 'OR':
					wires[out] = a or b
				elif op == 'XOR':
					wires[out] = (a or b) and not (a and b)
				todo[n] = False
				change = True
		return wires

	wires, circuit = s.split('\n\n')
	wires = {a[:a.index(':')]: int(a[a.index(' ') + 1:])
			for a in wires.splitlines()}
	circuit = [line.split() for line in circuit.splitlines()]
	newwires = run(wires.copy(), circuit)
	zwires = [a for a in sorted(newwires) if a[0] == 'z']
	result1 = sum((1 << n) * newwires[a] for n, a in enumerate(zwires))

	result2 = []
	switched = set()
	xorred = {x for a, op, b, _, out in circuit if op == 'XOR'
			for x in (a, b)}
	orred = {x for a, op, b, _, out in circuit if op == 'OR'
			for x in (a, b)}
	anded = {x for a, op, b, _, out in circuit if op == 'AND'
			for x in (a, b)}
	# hasinp = {}
	# for rule in circuit:
	# 	if rule[0] not in hasinp:
	# 		hasinp[rule[0]] = []
	# 	if rule[2] not in hasinp:
	# 		hasinp[rule[2]] = []
	# 	hasinp[rule[0]].append(rule)
	# 	hasinp[rule[2]].append(rule)
	# for n in range(45):
	# 	agenda = [f'x{n:02d}']
	# 	toprint = set()
	# 	for _ in range(3):
	# 		newagenda = []
	# 		for a in agenda:
	# 			toprint.update(tuple(x) for x in hasinp.get(a, ()))
	# 			for rule in hasinp.get(a, ()):
	# 				newagenda.append(rule[-1])
	# 		agenda = newagenda
	# 	for rule in sorted(toprint, key=lambda x: (
	# 			x[1] == 'OR', x[1] == 'AND', x[0][0] not in 'xy')):
	# 		print(' '.join(rule))
	# 	print()
	for a, op, b, _, out in circuit:
		if out[0] == 'z' and op != 'XOR' and out != 'z45':
			switched.add(out)
		if op == 'XOR' and out not in xorred | anded and out[0] != 'z':
			switched.add(out)
		if (op == 'XOR' and out[0] != 'z' and a[0] not in 'xy'
				and b[0] not in 'xy'):
			switched.add(out)
		if op == 'AND' and out not in orred and {a, b} != {'x00', 'y00'}:
			switched.add(out)
		if op == 'OR' and out not in xorred | anded and out != 'z45':
			switched.add(out)
	result2 = ','.join(sorted(switched))
	return result1, result2


def day25(s):
	keys, locks =  [[np.array([list(b) for b in a.splitlines()]) == '#'
			for a in s.split('\n\n') if a[0] == c]
			for c in '.#']
	return sum(not (lock & key).any()
			for lock in locks
				for key in keys)


if __name__ == '__main__':
	main(globals())
