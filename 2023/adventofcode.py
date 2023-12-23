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
		start = 0, 1
		end = len(grid) - 1, len(grid[0]) - 2
		ymax, xmax = len(grid), len(grid[0])
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

	grid = s.splitlines()
	path1 = f()
	# for y, line in enumerate(grid):
	# 	print(''.join('O' if (y, x) in path1 else c for x, c in enumerate(line)))
	return len(path1)



if __name__ == '__main__':
	main(globals())
