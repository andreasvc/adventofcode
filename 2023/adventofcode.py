"""Advent of Code 2023. http://adventofcode.com/2023 """
import re
import sys
from math import prod, lcm
# import json
import itertools
# from operator import lt, gt, eq
from functools import cache  # , reduce
# from collections import Counter, defaultdict
from heapq import heappush, heappop
from colorama import Fore, Style
import numpy as np
# from numba import njit
# from numba.typed import List
sys.path.append('..')
from common import main


def day1(s):
	digits = 'one two three four five six seven eight nine'.split()
	x = re.compile(r'\d|' + '|'.join(digits))
	xx = re.compile(r'\d|' + '|'.join(digits)[::-1])
	conv = dict(zip(digits, range(1, 10)))
	conv.update((str(a), a) for a in range(10))
	result1 = result2 = 0
	for line in s.splitlines():
		digits = re.findall(r'\d', line)
		result1 += int(digits[0] + digits[-1])
		digit1 = x.search(line).group()
		digit2 = xx.search(line[::-1]).group()[::-1]
		result2 += 10 * conv[digit1] + conv[digit2]
	return result1, result2


def day2(s):
	maxnum = {'red': 12, 'green': 13, 'blue': 14}
	pat = re.compile(r'(\d+) (red|green|blue)')
	result1 = result2 = 0
	for line in s.splitlines():
		gameid, rest = line.split(':', 1)
		minnum = {'red': 0, 'green': 0, 'blue': 0}
		possible = True
		for num, color in pat.findall(rest):
			num = int(num)
			if num > maxnum[color]:
				possible = False
			if num > minnum[color]:
				minnum[color] = num
		if possible:
			result1 += int(gameid.split()[1])
		result2 += prod(minnum.values())
	return result1, result2


def day3(s):
	grid = s.splitlines()
	symb = {(yy, xx)
			for y, line in enumerate(grid)
			for x, char in enumerate(line)
			if char != '.' and not char.isdigit()
			for yy, xx in itertools.product(
				(y - 1, y, y + 1), (x - 1, x, x + 1))}
	num = {(y, x): (int(match.group()), y, match.start())
			for y, line in enumerate(grid)
			for match in re.finditer(r'\d+', line)
			for x in range(match.start(), match.end())}
	result1 = result2 = 0
	for y, line in enumerate(grid):
		for x, char in enumerate(line):
			if char == '*':
				nums = {num[yy, xx]
						for yy, xx in itertools.product(
							(y - 1, y, y + 1), (x - 1, x, x + 1))
						if (yy, xx) in num}
				if len(nums) == 2:
					result2 += prod(n for n, _, _ in nums)
	result1 = sum(a for a, _, _ in {num[y, x] for y, x in symb & num.keys()})
	return result1, result2


def day4(s):
	result1 = 0
	lines = s.splitlines()
	cnt = [1 for _ in lines]
	for n, line in enumerate(lines):
		wins, ours = line.split(':')[1].split('|')
		wins = len({int(a) for a in wins.split()}
				& {int(a) for a in ours.split()})
		if wins:
			result1 += 2 ** (wins - 1)
		for m in range(n + 1, n + wins + 1):
			cnt[m] += cnt[n]
	return result1, sum(cnt)


def day5(s):
	def _day5(n, maps):
		for m in maps:
			for a, b, c in m:
				if b <= n < b + c:
					n = a + n - b
					break
		return n

	maps = s.split('\n\n')
	seeds = [int(a) for a in maps[0].split(':')[1].split()]
	maps = [[[int(a) for a in line.split()]
				for line in m.splitlines()[1:]]
			for m in maps[1:]]
	result2 = _day5(seeds[0], maps)
	for a, b in zip(seeds[::2], seeds[1::2]):
		for n in range(a, a + b, max(1, b // 100)):
			x = _day5(n, maps)
			if x < result2:
				result2, ma, mn = x, a, n
	for m in range(10, -1, -1):
		step = 2 ** m
		while mn - step >= ma:
			x = _day5(mn - step, maps)
			if x < result2:
				result2, mn = x, mn - step
			else:
				break
	return min(_day5(s, maps) for s in seeds), result2


def day6(s):
	def f(a, b):
		return sum((a - n) * n > b for n in range(1, a + 1))

	def search(a, b, f):
		lo, hi = 0, a
		while lo < hi:
			mid = (lo + hi) // 2
			if f(mid):
				lo = mid + 1
			else:
				hi = mid
		return lo

	times, dists = [line.split(':')[1].split() for line in s.splitlines()]
	result1 = prod(f(a, b) for a, b in zip(map(int, times), map(int, dists)))
	a, b = int(''.join(times)), int(''.join(dists))
	hi = search(a, b, lambda n: (a - n) * n > b)
	lo = search(a, b, lambda n: (a - n) * n <= b)
	result2 = hi - lo
	return result1, result2


def day7(s):
	def evaluate_joker(hand):
		return max(evaluate(hand.replace('A', n))
				for n in 'MLKJIHGFEDCBA')

	def evaluate(hand):
		return sum(hand.count(a) for a in hand)

	lines = [line.split() for line in s.splitlines()]
	trans = str.maketrans('AKQJT98765432', 'MLKJIHGFEDCBA')
	hands = [[hand.translate(trans), int(bid)] for hand, bid in lines]
	result1 = sum(rank * bid for rank, (_hand, bid)
			in enumerate(sorted(hands, key=lambda x: (evaluate(x[0]), x[0])), 1))
	trans = str.maketrans('AKQT98765432J', 'MLKJIHGFEDCBA')
	hands = [[hand.translate(trans), int(bid)] for hand, bid in lines]
	result2 = sum(rank * bid for rank, (_hand, bid)
			in enumerate(sorted(hands, key=lambda x: (evaluate_joker(x[0]), x[0])), 1))
	return result1, result2


def day8(s):
	lines = s.splitlines()
	dirs = lines[0]
	graph = {a: (b, c) for a, b, c
			in [re.findall(r'\w\w\w', line) for line in lines[2:]]}
	dirs = itertools.cycle(dirs)
	node = 'AAA'
	result1 = 0
	while node != 'ZZZ':
		node = graph[node][next(dirs) == 'R']
		result1 += 1
	nodes = [a for a in graph if a.endswith('A')]
	firstz = {}
	steps = 0
	while len(firstz) != len(nodes):
		d = next(dirs)
		nodes = [graph[node][d == 'R'] for node in nodes]
		steps += 1
		for n, node in enumerate(nodes):
			if n not in firstz and node.endswith('Z'):
				firstz[n] = steps
	result2 = lcm(*firstz.values())
	return result1, result2


def day9(s):
	result1 = result2 = 0
	for line in s.splitlines():
		nums = [[int(a) for a in line.split()]]
		while any(nums[-1]):
			nums.append([a - b for a, b in zip(nums[-1][1:], nums[-1])])
		nums[-1].append(0)
		for a, b in zip(nums[:-1][::-1], nums[::-1]):
			a.append(a[-1] + b[-1])
			a.insert(0, a[0] - b[0])
		result1 += nums[0][-1]
		result2 += nums[0][0]
	return result1, result2


def day10(s):
	def dump():
		for yy, line in enumerate(grid):
			print(''.join(
				Fore.GREEN + unilines.get(b, b) + Style.RESET_ALL
				if (yy, xx) in visited
				else Fore.RED + unilines.get(b, b) + Style.RESET_ALL
				if (yy * r + 1, xx * r + 1) in outside
				else unilines.get(b, b) for xx, b in enumerate(line)))
		print()

	def dump2():
		for yy, line in enumerate(supergrid):
			print(''.join(
				Fore.GREEN + unilines.get(b, b) + Style.RESET_ALL
				if (yy, xx) in svisited
				else Fore.RED + unilines.get(b, b) + Style.RESET_ALL
				if (yy, xx) in outside
				else unilines.get(b, b) for xx, b in enumerate(line)))
		print()

	unilines = {'|': '│', '-': '─', 'L': '└', 'J': '┘', '7': '┐', 'F': '┌'}
	grid = ['.' + line + '.' for line in s.splitlines()]
	grid = ['.' * len(grid[0])] + grid + ['.' * len(grid[0])]
	conn = {'|': [(-1, 0), (1, 0)],
			'-': [(0, -1), (0, 1)],
			'L': [(-1, 0), (0, 1)],
			'J': [(-1, 0), (0, -1)],
			'7': [(1, 0), (0, -1)],
			'F': [(1, 0), (0, 1)]}
	y = [n for n, line in enumerate(grid) if 'S' in line][0]
	x = grid[y].index('S')
	visited = {(y, x)}
	options = [(y + dy, x + dx)
			for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
			if any(dy + ddy == dx + ddx == 0
				for ddy, ddx in conn[grid[y + dy][x + dx]])]
	while options:
		y, x = options.pop()
		visited.add((y, x))
		options = [(y + dy, x + dx) for dy, dx
				in conn[grid[y][x]]
				if (y + dy, x + dx) not in visited]
	result1 = (len(visited) // 2) + (len(visited) % 2)

	# double resolution of grid to enable flood fill between adjacent lines
	enlarge = [
			{'|': '.|', '-': '..', 'L': '.|', 'J': '.|', '7': '..', 'F': '..', '.': '..', 'S': '.|'},
			{'|': '.|', '-': '--', 'L': '.L', 'J': '-J', '7': '-7', 'F': '.F', '.': '..', 'S': '-S'}]
	r = 2
	supergrid = [''.join(enlarge[n][a] for a in line)
			for line in grid
				for n in range(r)]
	svisited = {(r * y + dy, r * x + dx) for y, x in visited
			for dy in range(r) for dx in range(r)
			if enlarge[dy][grid[y][x]][dx] != '.'}
	outside = svisited.copy()
	queue = [(0, 0)]
	while queue:
		y, x = queue.pop()
		if (y, x) not in outside:
			outside.add((y, x))
			for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				if (0 <= y + dy < len(supergrid)
						and 0 <= x + dx < len(supergrid[0])):
					queue.append((y + dy, x + dx))
	# dump()
	# dump2()
	result2 = len([(y, x) for y, _ in enumerate(grid)
				for x, _ in enumerate(grid[0])
				if (y * r + 1, x * r + 1) not in outside])
	return result1, result2


def day11(s):
	grid = [[a == '#' for a in line] for line in s.splitlines()]
	emptyrows = [n for n, line in enumerate(grid) if not any(line)]
	emptycols = [n for n, _ in enumerate(grid[0])
			if not any(line[n] for line in grid)]
	coords = [(y, x) for y, _ in enumerate(grid)
			for x, _ in enumerate(grid[0]) if grid[y][x]]
	result1 = result2 = 0
	for n, (y1, x1) in enumerate(coords):
		for y2, x2 in coords[n + 1:]:
			dist = abs(y2 - y1) + abs(x2 - x1)
			extrarows = sum(1 for n in emptyrows if y1 < n < y2)
			extracols = sum(1 for n in emptycols if x1 < n < x2)
			result1 += dist + extrarows + extracols
			result2 += dist + (1000000 - 1) * (extrarows + extracols)
	return result1, result2


def day12(s):
	@cache
	def f(line, nums):
		if len(nums) == 0:
			return 0 if '#' in line else 1
		elif len(line) == 0:
			return 0
		elif line[0] == '.':
			return f(line[1:], nums)
		elif line[0] == '#':
			if '.' in line[:nums[0]] or len(line) < nums[0] or (
					nums[0] < len(line) and line[nums[0]] == '#'):
				return 0
			return f(line[nums[0] + 1:], nums[1:])
		elif line[0] == '?':
			return f('#' + line[1:], nums) + f(line[1:], nums)

	result1 = result2 = 0
	for line in s.splitlines():
		line, nums = line.split(' ')
		nums = tuple(int(a) for a in nums.split(','))
		result1 += f(line, nums)
		result2 += f('?'.join([line] * 5), nums * 5)
	return result1, result2


def day13(s):
	def f(grid, diff):
		for n in range(grid.shape[1] - (grid.shape[1] % 2), 1, -2):
			if (grid[:, :n] != np.flip(grid[:, :n], axis=1)).sum() == diff:
				return n // 2
			if (grid[:, -n:] != np.flip(grid[:, -n:], axis=1)).sum() == diff:
				return (grid.shape[1] - n) + n // 2
		return 0

	result1 = result2 = 0
	for grid in s.split('\n\n'):
		grid = np.array([[a == '#' for a in line]
				for line in grid.splitlines()], dtype=int)
		result1 += f(grid, 0) + f(grid.T, 0) * 100
		result2 += f(grid, 2) + f(grid.T, 2) * 100
	return result1, result2


def day14(s):
	def tilt(grid):
		for x, y in zip(*(grid == 'O').nonzero()):
			a = y - 1
			while a >= 0 and grid[x, a] == '.':
				a -= 1
			if grid[x, a + 1] == '.':
				grid[x, a + 1], grid[x, y] = grid[x, y], grid[x, a + 1]

	grid = np.array([list(line) for line in s.splitlines()]).T
	maxy = grid.shape[1]
	tilt(grid)
	coords = tuple(zip(*(grid == 'O').nonzero()))
	result1 = sum(maxy - y for _, y in coords)
	grid = np.array([list(line) for line in s.splitlines()]).T
	coords = tuple(zip(*(grid == 'O').nonzero()))
	seen = {}
	while coords not in seen:
		seen[coords] = len(seen)
		for _ in range(4):
			tilt(grid)
			grid = np.rot90(grid)
		coords = tuple(zip(*(grid == 'O').nonzero()))
	start = seen[coords]
	length = len(seen) - start
	idx = start + (1000000000 - start) % length
	coords = [c for c, n in seen.items() if n == idx].pop()
	result2 = sum(maxy - y for _, y in coords)
	return result1, result2


def day15(s):
	def h(x):
		val = 0
		for char in x:
			val = (val + ord(char)) * 17 % 256
		return val

	result1 = sum(h(x) for x in s.split(','))
	boxes = [[] for _ in range(256)]
	for step in s.split(','):
		if '=' in step:
			a, b = step.split('=')
			chain = boxes[h(a)]
			for x in chain:
				if x[0] == a:
					x[1] = b
					break
			else:
				chain.append([a, b])
		elif '-' in step:
			a = step.strip('-')
			chain = boxes[h(a)]
			chain[:] = [x for x in chain if x[0] != a]
		else:
			raise ValueError
	result2 = sum(n * m * int(b)
			for n, chain in enumerate(boxes, 1)
			for m, (_, b) in enumerate(chain, 1))
	return result1, result2


def day16(s):
	def f(pos, dir):
		seen = set()
		beams = [(pos, dir)]
		while beams:
			(y, x), (dy, dx) = beams.pop()
			y += dy
			x += dx
			pos, dir = (y, x), (dy, dx)
			if (0 <= y < len(grid) and 0 <= x < len(grid[0])
					and (pos, dir) not in seen):
				seen.add((pos, dir))
				if grid[y][x] == '.':
					beams.append((pos, dir))
				elif grid[y][x] == '/':
					beams.append((pos, refl1[dir]))
				elif grid[y][x] == '\\':
					beams.append((pos, refl2[dir]))
				elif grid[y][x] == '|':
					if dx == 0:
						beams.append((pos, dir))
					else:
						beams.extend([(pos, up), (pos, down)])
				elif grid[y][x] == '-':
					if dy == 0:
						beams.append((pos, dir))
					else:
						beams.extend([(pos, left), (pos, right)])
		return len({pos for pos, _ in seen})

	def sides():
		for y, _ in enumerate(grid):
			yield f((y, -1), right)
			yield f((y, len(grid[0])), left)
		for x, _ in enumerate(grid[0]):
			yield f((-1, x), down)
			yield f((len(grid), x), up)

	grid = s.splitlines()
	down, right, up, left = [(1, 0), (0, 1), (-1, 0), (0, -1)]
	refl1 = {right: up, up: right, left: down, down: left}  # /
	refl2 = {right: down, down: right, left: up, up: left}  # \
	return f((0, -1), (0, 1)), max(sides())


def day17(s):
	def f(minsteps, maxsteps):
		start = 0, 0
		end = len(grid) - 1, len(grid[0]) - 1
		ymax, xmax = len(grid), len(grid[0])
		agenda = [(end[0] + end[1], 0) + start + (1, 0),
				(end[0] + end[1], 0) + start + (0, 1)]
		seen = {start + (1, 0): end[0] + end[1],
				start + (0, 1): end[0] + end[1]}
		while agenda:
			est, cost, y, x, dy, dx = heappop(agenda)
			if (y, x) == end:
				return cost
			if seen[y, x, dy, dx] < est:
				continue
			for ddy, ddx in [(-dx, -dy), (dx, dy)]:
				ncost = cost
				for n in range(1, maxsteps + 1):
					ny, nx = y + ddy * n, x + ddx * n
					if 0 <= ny < ymax and 0 <= nx < xmax:
						ncost += grid[ny][nx]
						est = ncost + abs(end[0] - ny) + abs(end[1] - nx)
						if (n >= minsteps and est < seen.get(
								(ny, nx, ddy, ddx), 99999999)):
							heappush(agenda, (est, ncost, ny, nx, ddy, ddx))
							seen[ny, nx, ddy, ddx] = est

	grid = [[int(a) for a in line] for line in s.splitlines()]
	return f(1, 3), f(4, 10)


def day18(s):
	def getpoints(usecolor):
		y = x = boundary = 0
		points = [(y, x)]
		for line in s.splitlines():
			dir, num, color = line.split()
			if usecolor:
				num = int(color[2:7], base=16)
				dy, dx = dirs['RDLU'[int(color[7])]]
			else:
				num = int(num)
				dy, dx = dirs[dir]
			y, x = y + dy * num, x + dx * num
			points.append((y, x))
			boundary += num
		return points, boundary

	def getarea(points, boundary):
		area = sum((x1 * y2) - (x2 * y1)
				for (y1, x1), (y2, x2) in zip(points, points[1:] + points[:1]))
		return area // 2 + boundary // 2 + 1

	dirs = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
	result1 = getarea(*getpoints(False))
	result2 = getarea(*getpoints(True))
	return result1, result2


def day19(s):
	def numparts(rng, todo):
		rule = todo[0]
		if rule == 'R':
			return 0
		elif rule == 'A':
			return prod(c - b for b, c in rng.values())
		elif rule in rules:
			return numparts(rng, rules[rule])
		elif '<' in rule:
			var, rest = rule.split('<')
			val, wf = rest.split(':')
			return numparts({a: (b, int(val)) if a == var else (b, c)
						for a, (b, c) in rng.items()}, rules.get(wf, [wf])
					) + numparts({a: (int(val), c) if a == var else (b, c)
						for a, (b, c) in rng.items()}, todo[1:])
		elif '>' in rule:
			var, rest = rule.split('>')
			val, wf = rest.split(':')
			return numparts({a: (int(val) + 1, c) if a == var else (b, c)
						for a, (b, c) in rng.items()}, rules.get(wf, [wf])
					) + numparts({a: (b, int(val) + 1) if a == var else (b, c)
						for a, (b, c) in rng.items()}, todo[1:])

	rules, parts = s.split('\n\n')
	rules = {name: rest.split(',') for name, rest in
			(line.rstrip('}').split('{', 1) for line in rules.splitlines())}
	parts = [{a: int(b) for a, b
			in (a.split('=') for a in line.strip('{}').split(','))}
			for line in parts.splitlines()]
	result1 = 0
	for part in parts:
		wf = 'in'
		while wf not in 'AR':
			for rule in rules[wf]:
				wf = rule
				if '<' in rule:
					var, rest = rule.split('<')
					val, wf = rest.split(':')
					if part[var] < int(val):
						break
				elif '>' in rule:
					var, rest = rule.split('>')
					val, wf = rest.split(':')
					if part[var] > int(val):
						break
				elif rule in 'AR':
					break
			if wf == 'A':
				result1 += sum(part.values())
	result2 = numparts(dict.fromkeys('xmas', (1, 4001)), rules['in'])
	return result1, result2


def day20(s):
	from collections import deque
	config = {'output': [], 'rx': []}
	state = {'output': 0, 'rx': 0}
	incoming = {}
	conj = {}
	flipflops = set()
	for line in ['button -> broadcaster'] + s.splitlines():
		mod, conn = line.split(' -> ')
		mmod = mod.strip('%&')
		state[mmod] = 0
		if mod.startswith('%'):
			flipflops.add(mmod)
		elif mod.startswith('&'):
			conj[mmod] = None
		config[mmod] = conn.split(', ')
		for a in conn.split(', '):
			incoming.setdefault(a, []).append(mmod)
	for mod in conj:
		conj[mod] = dict.fromkeys(incoming[mod], 0)
	cnt = [0, 0]
	target = incoming[incoming['rx'][0]]
	idx = {}  # first time each of target mods outputs 0
	n = 0
	while len(idx) < 4:
		n += 1
		queue = deque([(0, 'button')])
		while queue:
			pulse, mod = queue.pop()
			if mod in conj:
				pulse = not all(conj[mod].values())
			else:
				if mod in flipflops:
					if pulse:
						continue
					else:
						state[mod] = not state[mod]
				pulse = state[mod]
			for a in config[mod]:
				queue.appendleft((pulse, a))
				if n <= 1000:
					cnt[pulse] += 1
				if a in conj:
					conj[a][mod] = pulse
			for a in target:
				if n > 1 and not any(conj[a].values()) and a not in idx:
					idx[a] = n
	return prod(cnt), lcm(*idx.values())


def day21(s):
	from scipy.interpolate import lagrange
	grid = s.splitlines()
	ymax, xmax = len(grid), len(grid[0])
	for n, line in enumerate(grid):
		if 'S' in line:
			sy, sx = n, line.index('S')
			grid[n] = line.replace('S', '.')
	queue = {(sy, sx)}
	ys = [0]
	for _ in range(sy + 2 * ymax):
		newqueue = set()
		for y, x in queue:
			for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				if grid[(y + dy) % ymax][(x + dx) % xmax] == '.':
					newqueue.add((y + dy, x + dx))
		queue = newqueue
		ys.append(len(queue))
	xs = [sy, sy + ymax, sy + 2 * ymax]
	p = lagrange(range(3), [ys[x] for x in xs])
	return ys[64], round(p((26501365 - sy) // ymax))


def day22(s):
	def gravity(bricks):
		grid = np.zeros((xmax + 1, ymax + 1, zmax + 1), dtype=int)
		newbricks = []
		fallen = 0
		for n, (x1, y1, z1, x2, y2, z2) in enumerate(bricks, 1):
			if z1 == 1:
				grid[x1:x2 + 1, y1:y2 + 1, z1:z2 + 1] = n
				newbricks.append((x1, y1, z1, x2, y2, z2))
			else:
				nz = z1
				zd = z2 - z1 + 1
				while nz > 0 and not grid[
						x1:x2 + 1, y1:y2 + 1, nz:nz + zd].any():
					nz -= 1
				grid[x1:x2 + 1, y1:y2 + 1, nz + 1:nz + zd + 1] = n
				newbricks.append((x1, y1, nz + 1, x2, y2, nz + zd))
				fallen += nz + 1 != z1
		return newbricks, grid, fallen

	bricks = [[int(a) for a in re.findall(r'\d+', line)]
			for line in s.splitlines()]
	xmax = ymax = zmax = 0
	for x1, y1, z1, x2, y2, z2 in bricks:
		xmax = max(x1, x2, xmax)
		ymax = max(y1, y2, ymax)
		zmax = max(z1, z2, zmax)
	bricks = sorted(bricks, key=lambda x: x[2])
	newbricks, grid, fallen = gravity(bricks)
	supportedby = {0: set()}
	for n, (x1, y1, z1, x2, y2, z2) in enumerate(newbricks, 1):
		supportedby[n] = set(
				grid[x1:x2 + 1, y1:y2 + 1, z1 - 1].ravel()) - {0}
	removable = [n for n, (x1, y1, z1, x2, y2, z2) in enumerate(newbricks, 1)
			if not any(len(supportedby[a]) == 1 for a in
				grid[x1:x2 + 1, y1:y2 + 1, z2 + 1].ravel())]
	result1 = len(removable)
	result2 = 0
	for n in set(range(1, len(newbricks) + 1)) - set(removable):
		newbricks1 = [a for m, a in enumerate(newbricks, 1) if n != m]
		newbricks2, _grid, fallen = gravity(newbricks1)
		result2 += fallen
	return result1, result2


def day23(s):
	def f():
		agenda = [(set(), ) + start]
		hikes = []
		while agenda:
			seen, y, x = agenda.pop()
			if (y, x) == end:
				hikes.append(seen)
			for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				ny, nx = y + dy, x + dx
				if 0 <= ny < ymax and 0 <= nx < xmax:
					if grid[ny][nx] == '.' and (ny, nx) not in seen:
						agenda.append((seen | {(ny, nx)}, ny, nx))
					elif grid[ny][nx] in '<>^v' and (ny, nx) not in seen:
						ddy, ddx = {'<': (0, -1), '>': (0, 1),
								'^': (-1, 0), 'v': (1, 0)}[grid[ny][nx]]
						nny, nnx = ny + ddy, nx + ddx
						if (nny, nnx) not in seen:
							agenda.append((
									seen | {(ny, nx), (nny, nnx)}, nny, nnx))
		return max(hikes, key=len)

	def getdist(pos1, pos2):
		if pos1 == pos2:
			return -1
		agenda = [(0, ) + pos1]
		seen = set(nodes) - {pos2}
		while agenda:
			steps, y, x = agenda.pop()
			if (y, x) == pos2:
				return steps
			seen.add((y, x))
			for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				ny, nx = y + dy, x + dx
				if (0 <= ny < ymax and 0 <= nx < xmax
						and grid[ny][nx] == '.' and (ny, nx) not in seen):
					agenda.append((steps + 1, ny, nx))
		return -1

	@cache
	def g(start, end, unvisited):
		dists = []
		for pos, steps in graph[start]:
			if unvisited & pos:
				dist = 0 if pos == end else g(pos, end, unvisited ^ pos)
				if dist >= 0:
					dists.append(dist + steps)
		return max(dists, default=-9999999999)

	grid = s.splitlines()
	ymax, xmax = len(grid), len(grid[0])
	grid[0] = grid[0].replace('.', '#', 1)
	start = 0, 1
	end = len(grid) - 1, len(grid[0]) - 2
	path1 = f()

	grid = re.sub(r'[<>v^]', '.', s).splitlines()
	nodes = [start, end]
	for y in range(1, len(grid) - 1):
		for x in range(1, len(grid[0]) - 1):
			if (grid[y][x] + grid[y - 1][x] + grid[y + 1][x]
					+ grid[y][x - 1] + grid[y][x + 1]).count('.') > 3:
				nodes.append((y, x))
	bitnodes = {a: 1 << n for n, a in enumerate(nodes)}
	graph = {bitnodes[pos1]: [a for a in
				[(bitnodes[pos2], getdist(pos1, pos2)) for pos2 in nodes]
				if a[1] != -1]
			for pos1 in nodes}
	start, end = bitnodes[start], bitnodes[end]
	unvisited = (1 << len(nodes)) - 1
	path2 = g(start, end, unvisited ^ start)
	return len(path1), path2


def day24(s):
	# Source: https://stackoverflow.com/a/42727584
	def get_intersect(a1, a2, b1, b2):
		"""
		Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
		a1: [x, y] a point on the first line
		a2: [x, y] another point on the first line
		b1: [x, y] a point on the second line
		b2: [x, y] another point on the second line """
		s = np.vstack([a1, a2, b1, b2])     # s for stacked
		h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
		l1 = np.cross(h[0], h[1])           # get first line
		l2 = np.cross(h[2], h[3])           # get second line
		x, y, z = np.cross(l1, l2)          # point of intersection
		if z == 0:                          # lines are parallel
			return None
		return (x / z, y / z)

	def part1():
		result1 = 0
		for n in range(hail.shape[0]):
			for m in range(n + 1, hail.shape[0]):
				result = get_intersect(
						hail[n, [x, y]],
						hail[n, [x, y]] + 10000 * hail[n, [vx, vy]],
						hail[m, [x, y]],
						hail[m, [x, y]] + 10000 * hail[m, [vx, vy]])
				inbounds = (result is not None
						and bounds[0] <= result[0] <= bounds[1]
						and bounds[0] <= result[1] <= bounds[1])
				futa = (result is not None
						and (result[0] - hail[n, x]) / hail[n, vx] >= 0)
				futb = (result is not None
						and (result[0] - hail[m, x]) / hail[m, vx] >= 0)
				result1 += inbounds and futa and futb
				if hail.shape[0] == 5:
					print()
					print(n, s.splitlines()[n])
					print(m, s.splitlines()[m])
					print(inbounds and futa and futb, result, inbounds, futa, futb)
		return result1

	# Source: https://stackoverflow.com/a/18543221
	def isect_line_plane(p0, p1, p_co, p_no, epsilon=1e-6):
		"""
		p0, p1: Define the line.
		p_co, p_no: define the plane:
			p_co Is a point on the plane (plane coordinate).
			p_no Is a normal vector defining the plane direction;
				(does not need to be normalized).

		Return a Vector or None (when the intersection can't be found).
		"""
		u = p1 - p0
		dot = np.dot(p_no, u)

		if abs(dot) > epsilon:
			# The factor of the point between p0 -> p1 (0 - 1)
			# if 'fac' is between (0 - 1) the point intersects with the segment.
			# Otherwise:
			#  < 0.0: behind p0.
			#  > 1.0: infront of p1.
			w = p0 - p_co
			fac = -np.dot(p_no, w) / dot
			return p0 + u * fac

		# The segment is parallel to plane.
		return None

	hail = np.array(
			[[int(a) for a in re.findall(r'-?\d+', line)]
			for line in s.splitlines()], dtype=int)
	x, y, z, vx, vy, vz = range(6)
	bounds = [7, 27] if hail.shape[0] == 5 else	[
			200000000000000, 400000000000000]
	result1 = part1()

	# make all positions and velocities relative to stone 1
	h0 = hail[0, :].copy()
	hail -= h0
	mult = 1000000000  # to avoid under/overflow

	# plane defined by origin and line of stone 1
	# https://stackoverflow.com/a/53698872
	p0, p1, p2 = [hail[0, :3], hail[1, :3], hail[1, :3] + mult * hail[1, 3:]]
	# convert to Python integers to avoid overflow w/numpy?
	x0, y0, z0 = p0.tolist()
	x1, y1, z1 = p1.tolist()
	x2, y2, z2 = p2.tolist()
	ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]
	vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]
	u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
	point = np.array(p0)
	normal = np.array(u_cross_v)

	# intersect with line of stone 2
	p2 = isect_line_plane(
			hail[2, :3], hail[2, :3] + mult * hail[2, 3:],
			point, normal)
	# same for stone 3
	p3 = isect_line_plane(
			hail[3, :3], hail[3, :3] + mult * hail[3, 3:],
			point, normal)
	# now calculate time t2 for p2, and t3 for p3
	t2 = round((p2[2] - hail[2, 2]) / (hail[2, 5]))
	t3 = round((p3[2] - hail[3, 2]) / (hail[3, 5]))
	# velocity for rock (should be integers, so round)
	v = ((p2 - p3) / (t2 - t3)).round()
	# calculate position of rock at t0
	p = p2 - (t2 * v)
	# get back to normal coordinates (not relative to stone 1)
	p += h0[:3]
	result2 = round(p.sum())
	return result1, result2


def day25(s):
	import networkx as nx
	graph = nx.Graph()
	for line in s.splitlines():
		a, rest = line.split(':')
		graph.add_node(a)
		graph.add_nodes_from(rest.split())
	for line in s.splitlines():
		a, rest = line.split(':')
		for b in rest.split():
			graph.add_edge(a, b)
	for a in nx.community.girvan_newman(graph):
		return prod(len(b) for b in a)

if __name__ == '__main__':
	main(globals())
