"""Advent of Code 2021. http://adventofcode.com/2021 """
import re
import sys
import json
from operator import lt, gt, eq
from functools import reduce
import itertools
from collections import Counter, defaultdict
import numpy as np
# from numba import njit
sys.path.append('..')
from common import main


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


def day4a(s):
	chunks = s.split('\n\n')
	nums = [int(a) for a in chunks[0].split(',')]
	boards = [
		np.array([[int(a) for a in line.split()]
			for line in chunk.splitlines()], dtype=int)
		for chunk in chunks[1:]]
	drawn = set()
	while True:
		num = nums.pop(0)
		drawn.add(num)
		for board in boards:
			if any(set(board[n, :]) <= drawn
					or set(board[:, n]) <= drawn
					for n in range(5)):
				unmarked = set(board.flatten()) - drawn
				return sum(unmarked) * num


def day4b(s):
	chunks = s.split('\n\n')
	nums = [int(a) for a in chunks[0].split(',')]
	boards = [
		np.array([[int(a) for a in line.split()]
			for line in chunk.splitlines()], dtype=int)
		for chunk in chunks[1:]]
	drawn = set()
	while True:
		num = nums.pop(0)
		drawn.add(num)
		for n, board in enumerate(boards):
			if board is None:
				continue
			if any(set(board[n, :]) <= drawn
					or set(board[:, n]) <= drawn
					for n in range(5)):
				if sum(b is not None for b in boards) == 1:
					unmarked = set(board.flatten()) - drawn
					return sum(unmarked) * num
				boards[n] = None


def _day5(s, diagonals=True):
	lines = np.array([[int(a) for a in re.findall(r'\d+', line)]
			for line in s.splitlines()], dtype=int)
	size = lines.flatten().max() + 1
	grid = np.zeros((size, size), dtype=int)
	for n in range(lines.shape[0]):
		x1, y1, x2, y2 = lines[n, :]
		if diagonals or x1 == x2 or y1 == y2:
			xd = 0 if x1 == x2 else 1 if x2 > x1 else - 1
			yd = 0 if y1 == y2 else 1 if y2 > y1 else - 1
			for m in range(max(abs(x2 - x1), abs(y2 - y1)) + 1):
				grid[y1 + yd * m, x1 + xd * m] += 1
	return (grid >= 2).flatten().sum()


def day5a(s):
	return _day5(s, diagonals=False)


def day5b(s):
	return _day5(s, diagonals=True)


def _day6(s, days=80):
	x = [0] * 9
	for a in [int(a) for a in s.split(',')]:
		x[a] += 1
	for _ in range(days):
		x = x[1:] + x[:1]
		x[6] += x[8]
	return sum(x)


def day6a(s):
	return _day6(s, 80)


def day6b(s):
	return _day6(s, 256)


def day7a(s):
	nums = np.array([int(a) for a in s.split(',')])
	return min([
		np.abs(nums - n).sum()
		for n in range(min(nums), max(nums) + 1)])


def day7b(s):
	def triangle(x):
		return x * (x + 1) // 2

	nums = np.array([int(a) for a in s.split(',')])
	mean = nums.mean()
	return min([
		triangle(np.abs(nums - n)).sum()
		for n in (round(mean - 0.5), round(mean + 0.5))])


def day8a(s):
	lines = [[a.split() for a in line.split(' | ')]
			for line in s.splitlines()]
	return sum(1
			for line in lines
			for a in line[1]
			if len(a) in (2, 3, 4, 7))


def day8b(s):
	lines = [[[frozenset(x) for x in a.split()]
			for a in line.split(' | ')]
			for line in s.splitlines()]
	digits = {   # segments                num missing
			1: set('  c  f ') - set(' '),  # 2 abdeg
			7: set('a c  f ') - set(' '),  # 3 bdeg
			4: set(' bcd f ') - set(' '),  # 4 aeg
			2: set('a cde g') - set(' '),  # 5 bf
			3: set('a cd fg') - set(' '),  # 5 be
			5: set('ab d fg') - set(' '),  # 5 ce
			6: set('ab defg') - set(' '),  # 6 c
			0: set('abc efg') - set(' '),  # 6 d
			9: set('abcd fg') - set(' '),  # 6 e
			8: set('abcdefg') - set(' '),  # 7 {}
			#       8687497  <-- segment counts
			}
	lenmap = {2: 1, 3: 7, 4: 4, 7: 8}  # lengths that map to unique digits
	countmap = {4: 'e', 6: 'b', 9: 'f'}  # unique segment counts -> segments
	result = 0
	for signals, output in lines:
		# possible mappings for each letter: original -> possible new letters
		poss = {letter: set('abcdefg') for letter in 'abcdefg'}
		cnt = Counter()
		for signal in signals:
			cnt.update(signal)
			if len(signal) in lenmap:
				d = lenmap[len(signal)]
				for letter in digits[d]:
					poss[letter] &= signal
				for letter in poss.keys() - digits[d]:
						poss[letter] -= signal
		for letter, c in cnt.items():
			if c in countmap:
				newletter = countmap[c]
				poss[newletter] = {letter}
				for a in poss:
					if a != newletter:
						poss[a] -= {letter}
		assert all(len(a) == 1 for a in poss.values())
		poss = {a: c for a, b in poss.items() for c in b}
		mapping = {frozenset(poss[a] for a in letters): d
				for d, letters in digits.items()}
		result += int(''.join(str(mapping[a]) for a in output))
	return result


def day9a(s):
	grid = np.array([[int(a) for a in line] for line in s.splitlines()],
			dtype=int)
	ymax, xmax = grid.shape
	result = 0
	for y in range(ymax):
		for x in range(xmax):
			if all(grid[y, x] < grid[y + yd, x + xd]
					for yd, xd in ((-1, 0), (0, 1), (0, -1), (1, 0))
					if 0 <= y + yd < ymax and 0 <= x + xd < xmax):
				result += grid[y, x] + 1
	return result


def day9b(s):
	from scipy.cluster.hierarchy import DisjointSet
	grid = np.array([[int(a) for a in line] for line in s.splitlines()],
			dtype=int)
	ymax, xmax = grid.shape
	loc = [(y, x) for y in range(ymax)
			for x in range(xmax)
			if grid[y, x] < 9]
	basins = DisjointSet(loc)
	for y, x in loc:
		neighbors = [(grid[y + yd, x + xd], y + yd, x + xd)
				for yd, xd in ((-1, 0), (0, 1), (0, -1), (1, 0))
				if 0 <= y + yd < ymax and 0 <= x + xd < xmax
					and grid[y, x] < 9]
		val, yy, xx = min(neighbors, default=(999, -1, -1))
		if grid[y, x] > val:
			basins.merge((y, x), (yy, xx))
	a, b, c = sorted(basins.subsets(), key=len)[-3:]
	return len(a) * len(b) * len(c)


def day10a(s):
	stack = []
	mapping = dict(zip('([{<', ')]}>'))
	scores = dict(zip(')]}>', [3, 57, 1197, 25137]))
	result = 0
	for line in s.splitlines():
		for n, char in enumerate(line):
			if char in '([{<':
				stack.append(char)
			elif char in ')]}>':
				if stack and mapping[stack[-1]] == char:
					stack.pop()
				else:
					result += scores[char]
					break
	return result


def day10b(s):
	mapping = dict(zip('([{<', ')]}>'))
	scores = dict(zip(')]}>', [1, 2, 3, 4]))
	results = []
	for line in s.splitlines():
		stack = []
		for n, char in enumerate(line):
			if char in '([{<':
				stack.append(char)
			elif char in ')]}>':
				if stack and mapping[stack[-1]] == char:
					stack.pop()
				else:
					break
		else:
			result = 0
			for char in stack[::-1]:
				result *= 5
				result += scores[mapping[char]]
			results.append(result)
	return sorted(results)[len(results) // 2]


def _day11(s):
	grid = np.array([[int(a) for a in line]
			for line in s.splitlines()], dtype=int)
	numflashes = 0
	while grid.any():
		grid += 1
		flashing = list(zip(*(grid > 9).nonzero()))
		flashed = set(flashing)
		while flashing:
			x, y = flashing.pop()
			numflashes += 1
			x1, x2 = max(x - 1, 0), min(grid.shape[0], x + 2)
			y1, y2 = max(y - 1, 0), min(grid.shape[1], y + 2)
			grid[x1:x2, y1:y2] += 1
			newflashes = set(zip(*(grid > 9).nonzero())) - flashed
			flashed.update(newflashes)
			flashing.extend(newflashes)
		grid[grid > 9] = 0
		yield numflashes, grid


def day11a(s):
	numflashes, _ = next(itertools.islice(_day11(s), 99, None))
	return numflashes


def day11b(s):
	for n, (_, grid) in enumerate(_day11(s), 1):
		if not grid.any().any():
			return n


def _day12(s, twice=True):
	graph = defaultdict(list)
	for line in s.splitlines():
		a, b = line.split('-')
		if b != 'start':
			graph[a].append(b)
		if a != 'start':
			graph[b].append(a)
	agenda = [('start', set(), twice)]
	result = 0
	while agenda:
		pos, route, twice = agenda.pop()
		for a in graph[pos]:
			if a == 'end':
				result += 1
			elif a.isupper():
				agenda.append((a, route, twice))
			elif a not in route:
				agenda.append((a, route | {a}, twice))
			elif not twice:
				agenda.append((a, route, True))
	return result


def day12a(s):
	return _day12(s)


def day12b(s):
	return _day12(s, twice=False)


def _day13(s, firstonly=True):
	fst, sec = s.split('\n\n')
	xcoords = [int(a.split(',')[0]) for a in fst.splitlines()]
	ycoords = [int(a.split(',')[1]) for a in fst.splitlines()]
	grid = np.zeros((max(ycoords) + 1, max(xcoords) + 1), dtype=int)
	grid[ycoords, xcoords] = 1
	for fold in sec.splitlines():
		axis, pos = fold.split()[2].split('=')
		pos = int(pos)
		if axis == 'x':
			grid = grid[:, :pos] + np.fliplr(grid[:, pos + 1:])
		elif axis == 'y':
			grid = grid[:pos, :] + np.flipud(grid[pos + 1:, :])
		if firstonly:
			return (grid !=0).sum()
	return '\n'.join(''.join('#' if grid[y, x] else ' '
				for x in range(grid.shape[1]))
			for y in range(grid.shape[0]))


def day13a(s):
	return _day13(s)


def day13b(s):
	return _day13(s, firstonly=False)


def day14a(s):
	template, rules = s.split('\n\n')
	rules = dict(a.split(' -> ') for a in rules.splitlines())
	x = template
	for n in range(10):
		new = ''
		for a, b in zip(x, x[1:]):
			if a + b in rules:
				new += a + rules[a + b]
			else:
				new += a
		new += x[-1]
		x = new
	cnt = Counter(x)
	x = cnt.most_common()
	return x[0][1] - x[-1][1]


def day14b(s):
	template, rules = s.split('\n\n')
	rules = {tuple(a.split(' -> ')[0]): a.split(' -> ')[1]
			for a in rules.splitlines()}
	state = Counter(zip(template, template[1:]))
	cnt = Counter(template)
	for n in range(40):
		new = Counter()
		for ab in list(state):
			if ab in rules:
				x = state.pop(ab)
				c = rules[ab]
				new[ab[0], c] += x
				new[c, ab[1]] += x
				cnt[c] += x
		state.update(new)
	x = cnt.most_common()
	return x[0][1] - x[-1][1]


def _day15(dists):
	agenda = [[0, 0, 0]] + [[] for _ in range(9)]
	seen = np.zeros((dists.shape[0] + 1, dists.shape[1] + 1), dtype=bool)
	seen[:, -1] = seen[-1, :] = True
	endy, endx = dists.shape[0] - 1, dists.shape[1] - 1
	for ag in itertools.cycle(agenda):
		while ag:
			cost, y, x = ag.pop(), ag.pop(), ag.pop()
			if x == endx and y == endy:
				return cost
			newx = x + 1
			if not seen[y, newx]:
				seen[y, newx] = True
				newcost = cost + dists[y, newx]
				agenda[newcost % 10].extend([newx, y, newcost])
			newx -= 2
			if not seen[y, newx]:
				seen[y, newx] = True
				newcost = cost + dists[y, newx]
				agenda[newcost % 10].extend([newx, y, newcost])
			newy = y + 1
			if not seen[newy, x]:
				seen[newy, x] = True
				newcost = cost + dists[newy, x]
				agenda[newcost % 10].extend([x, newy, newcost])
			newy -= 2
			if not seen[newy, x]:
				seen[newy, x] = True
				newcost = cost + dists[newy, x]
				agenda[newcost % 10].extend([x, newy, newcost])


def day15a(s):
	dists = np.array([[int(dist) for x, dist in enumerate(line)]
			for y, line in enumerate(s.splitlines())], dtype=int)
	return _day15(dists)


def day15b(s):
	dists = [[int(dist) for x, dist in enumerate(line)]
			for y, line in enumerate(s.splitlines())]
	dists = np.array(dists, dtype=int)
	endy, endx = dists.shape
	X = np.zeros((endy * 5, endx * 5), dtype=int)
	for row in range(5):
		for col in range(5):
			n = row + col
			X[row * endy:(row + 1) * endy,
					col * endx:(col + 1) * endx] = (dists + n - 1) % 9 + 1
	return _day15(X)


def prod(seq, start=1):
	for a in seq:
		start *= a
	return start


def _day16(x):
	version, typeid = int(x[:3], 2), int(x[3:6], 2)
	result, n = version, 6
	if typeid == 4:  # literal value
		tmp = ''
		while x[n] == '1':
			tmp += x[n + 1:n + 5]
			n += 5
		tmp += x[n + 1:n + 5]
		val = int(tmp, 2)
		n += 5
		return n, result, val
	elif x[n] == '0':  # n bits of subpackets
		goal = n + 16 + int(x[n + 1:n + 16], 2)
		n += 16
		vals = []
		while n < goal:
			nn, rresult, val = _day16(x[n:])
			n += nn
			result += rresult
			vals.append(val)
	elif x[n] == '1':  # n subpackets
		n += 12
		vals = []
		for _ in range(int(x[n - 11:n], 2)):
			nn, rresult, val = _day16(x[n:])
			n += nn
			result += rresult
			vals.append(val)
	arrops, binops = [sum, prod, min, max, int], [gt, lt, eq]
	val = arrops[typeid](vals) if typeid < 5 else binops[typeid - 5](*vals)
	return n, result, val


def day16a(s):
	"""
	>>> day16a('8A004A801A8002F478')
	16
	>>> day16a('620080001611562C8802118E34')
	12
	>>> day16a('C0015000016115A2E0802F182340')
	23
	>>> day16a('A0016C880162017C3686B18A3D4780')
	31"""
	x = ''.join(bin(int(a, 16))[2:].zfill(4) for a in s.strip())
	return _day16(x)[1]


def day16b(s):
	"""
	>>> day16b('C200B40A82')
	3
	>>> day16b('04005AC33890')
	54
	>>> day16b('880086C3E88112')
	7
	>>> day16b('CE00C43D881120')
	9
	>>> day16b('D8005AC2A8F0')
	1
	>>> day16b('F600BC2D8F')
	0
	>>> day16b('9C005AC2F8F0')
	0
	>>> day16b('9C0141080250320F1802104A08')
	1"""
	x = ''.join(bin(int(a, 16))[2:].zfill(4) for a in s.strip())
	return _day16(x)[2]


# @njit(cache=True)
def _day17(x1, x2, y1, y2):
	yymax = 0
	cnt = 0
	for velx in range(x2 + 1):
		for vy in range(-abs(y1), abs(y1)):
			vx = velx
			x = y = ymax = 0
			while True:
				x += vx
				y += vy
				vx -= (vx > 0) - (vx < 0)
				vy -= 1
				if y > ymax:
					ymax = y
				if x1 <= x <= x2 and y1 <= y <= y2:
					cnt += 1
					if ymax > yymax:
						yymax = ymax
					break
				elif y < 2 * y2 or x > 2 * x2:
					break
	return yymax, cnt


def day17a(s):
	x1, x2, y1, y2 = [int(a) for a in re.findall(r'-?\d+', s)]
	return _day17(x1, x2, y1, y2)[0]


def day17b(s):
	x1, x2, y1, y2 = [int(a) for a in re.findall(r'-?\d+', s)]
	return _day17(x1, x2, y1, y2)[1]


def add(a, b):
	"""
	>>> add([[[[4,3],4],4],[7,[[8,4],9]]], [1,1])
	[[[[0, 7], 4], [[7, 8], [6, 0]]], [8, 1]]
	>>> add([[[[4,0],[5,4]],[[7,7],[6,0]]],[[8,[7,7]],[[7,9],[5,0]]]],
	...			[[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]])
	[[[[6, 7], [6, 7]], [[7, 7], [0, 7]]], [[[8, 7], [7, 7]], [[8, 8], [8, 0]]]]
	>>> add([[[[7,0],[7,7]],[[7,7],[7,8]]],[[[7,7],[8,8]],[[7,7],[8,7]]]],
	...		[7,[5,[[3,8],[1,4]]]])
	[[[[7, 7], [7, 8]], [[9, 5], [8, 7]]], [[[6, 8], [0, 8]], [[9, 9], [9, 0]]]]
	"""
	result = [a, b]
	while doexplode(result, 0, 4)[0] or dosplit(result, 10):
		pass
	return result


def doexplode(node, cur, maxdepth, digits=re.compile(r'\d+')):
	"""Find leftmost node nested at maxdepth and explode it.

	>>> tree = [[[[[9, 8], 1], 2], 3], 4]
	>>> _ = doexplode(tree,  0,  4)
	>>> tree
	[[[[0, 9], 2], 3], 4]
	>>> tree = [7, [6, [5, [4, [3, 2]]]]]
	>>> _ = doexplode(tree,  0,  4)
	>>> tree
	[7, [6, [5, [7, 0]]]]
	>>> tree = [[6, [5, [4, [3, 2]]]], 1]
	>>> _ = doexplode(tree,  0,  4)
	>>> tree
	[[6, [5, [7, 0]]], 3]
	>>> tree = [[3, [2, [1, [7, 3]]]], [6, [5, [4, [3, 2]]]]]
	>>> _ = doexplode(tree,  0,  4)
	>>> tree
	[[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]]
	>>> tree = [[3, [2, [8, 0]]], [9, [5, [4, [3, 2]]]]]
	>>> _ = doexplode(tree,  0,  4)
	>>> tree
	[[3, [2, [8, 0]]], [9, [5, [7, 0]]]]"""
	def findsibling(node, x, right):
		if isinstance(node, int):
			return node + x
		return [node[0], findsibling(node[1], x, right)
				] if right else [findsibling(node[0], x, right), node[1]]

	l = r = None
	exploded = False
	for i, child in enumerate(node):
		if isinstance(child, int):
			continue
		elif cur + 1 >= maxdepth and all(isinstance(gc, int) for gc in child):
			l, r = child
			node[i] = 0
			exploded = True
		else:
			exploded, l, r = doexplode(child, cur + 1, maxdepth)
		if l is not None and i == 1:
			node[0] = findsibling(node[0], l, i)
			l = None
		if r is not None and i == 0:
			node[1] = findsibling(node[1], r, i)
			r = None
		if exploded:
			break
	return exploded, l, r


def dosplit(node, maxval):
	"""Find leftmost leaf with value >= maxval and split.
	>>> node = [11, 1]
	>>> dosplit(node, 10)
	True
	>>> node
	[[5, 6], 1]
	"""
	for i, child in enumerate(node):
		if isinstance(child, int):
			if child >= maxval:
				a = b = child // 2
				b += child & 1
				node[i] = [a, b]
				return True
		elif dosplit(child, maxval):
			return True
	return False


def magnitude(node):
	"""
	>>> magnitude([[1,2],[[3,4],5]])
	143
	>>> magnitude([[[[0,7],4],[[7,8],[6,0]]],[8,1]])
	1384
	>>> magnitude([[[[1,1],[2,2]],[3,3]],[4,4]])
	445
	>>> magnitude([[[[3,0],[5,3]],[4,4]],[5,5]])
	791
	>>> magnitude([[[[5,0],[7,4]],[5,5]],[6,6]])
	1137
	>>> magnitude([[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]])
	3488
	"""
	if isinstance(node, int):
		return node
	return 3 * magnitude(node[0]) + 2 * magnitude(node[1])


def day18a(s):
	"""
	>>> s = ('[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]\\n'
	...			'[[[5,[2,8]],4],[5,[[9,9],0]]]\\n'
	...			'[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]\\n'
	...			'[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]\\n'
	...			'[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]\\n'
	...			'[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]\\n'
	...			'[[[[5,4],[7,7]],8],[[8,3],8]]\\n'
	...			'[[9,3],[[9,9],[6,[4,9]]]]\\n'
	...			'[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]\\n'
	...			'[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]\\n')
	>>> day18a(s)
	4140
	>>> reduce(add, [[1, 1], [2, 2], [3, 3], [4, 4]])
	[[[[1, 1], [2, 2]], [3, 3]], [4, 4]]
	>>> reduce(add, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
	[[[[3, 0], [5, 3]], [4, 4]], [5, 5]]
	>>> reduce(add, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
	[[[[5, 0], [7, 4]], [5, 5]], [6, 6]]
	>>> data = [
	...		[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]],
	...		[7,[[[3,7],[4,3]],[[6,3],[8,8]]]],
	...		[[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]],
	...		[[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]],
	...		[7,[5,[[3,8],[1,4]]]],
	...		[[2,[2,2]],[8,[8,1]]],
	...		[2,9],
	...		[1,[[[9,3],9],[[9,0],[0,7]]]],
	...		[[[5,[7,4]],7],1],
	...		[[[[4,2],2],6],[8,7]]]
	>>> reduce(add, data)
	[[[[8, 7], [7, 7]], [[8, 6], [7, 7]]], [[[0, 7], [6, 6]], [8, 7]]]
	"""
	data = [json.loads(line) for line in s.splitlines()]
	return magnitude(reduce(add, data))


def day18b(s):
	"""
	>>> s = ('[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]\\n'
	...			'[[[5,[2,8]],4],[5,[[9,9],0]]]\\n'
	...			'[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]\\n'
	...			'[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]\\n'
	...			'[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]\\n'
	...			'[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]\\n'
	...			'[[[[5,4],[7,7]],8],[[8,3],8]]\\n'
	...			'[[9,3],[[9,9],[6,[4,9]]]]\\n'
	...			'[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]\\n'
	...			'[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]\\n')
	>>> day18b(s)
	3993
	"""
	return max(magnitude(add(json.loads(a), json.loads(b)))
			for a, b in itertools.permutations(s.splitlines(), 2))


def _day19(s):
	from scipy.spatial import distance
	scanners = [np.array([[int(a) for a in line.split(',')]
				for line in block.splitlines()[1:]], dtype=int)
			for block in s.split('\n\n')]
	# all possible rotations without mirroring
	# https://github.com/xdavidliu/advent-code-2021/blob/main/day19.py
	rot = []
	for x, y, z in itertools.product(*[(1, -1)] * 3):
		for q in itertools.permutations([[x, 0, 0], [0, y, 0], [0, 0, z]]):
			m = np.array(q)
			if np.linalg.det(m) == 1:
				rot.append(m)
	aligned = {0: scanners[0]}
	loc = {0: np.array([0, 0, 0], dtype=int)}
	while len(aligned) < len(scanners):
		for n, scanner in enumerate(scanners):
			if n in aligned:
				continue
			for r in rot:
				x = scanner @ r
				for other in aligned.values():
					res = distance.cdist(other, x, 'euclidean')
					vals, counts = np.unique(res.ravel(), return_counts=True)
					index = np.argmax(counts)
					val, cnt = vals[index], counts[index]
					if cnt >= 12:
						ac, bc = (res == val).nonzero()
						diff = other[ac[0], :] - x[bc[0], :]
						loc[n] = diff
						aligned[n] = x + diff
						break
				if n in aligned:
					break
	return np.vstack(list(aligned.values())), loc


def day19a(s):
	aligned, _loc = _day19(s)
	unique = np.unique(aligned, axis=0)
	return unique.shape[0]


def day19b(s):
	_aligned, loc = _day19(s)
	return max(np.abs(a - b).sum()
			for a in loc.values()
			for b in loc.values())


def day20a(s, steps=2):
	def show(im):
		print('\n'.join(
				''.join('#' if im[n, m] else '.' for m in range(im.shape[1]))
				for n in range(im.shape[0])) + '\n')

	from scipy.ndimage import convolve
	alg, im = s.split('\n\n')
	alg = np.array([a == '#' for a in alg], dtype=int)
	im = np.array([[a == '#' for a in line]
			for line in im.splitlines()], dtype=int)
	im = np.pad(im, steps + 1)
	conv = (2 ** np.arange(9, dtype=int)).reshape((3, 3))
	for _ in range(steps):
		im = alg[convolve(im, conv)]
	return int(im.sum())


def day20b(s):
	return day20a(s, steps=50)


def day21a(s):
	_, a, _, b = map(int, re.findall(r'\d+', s))
	ascore = bscore = cnt = 0
	die = itertools.cycle(range(1, 101))
	while True:
		a = (a - 1 + next(die) + next(die) + next(die)) % 10 + 1
		ascore += a
		cnt += 3
		if ascore >= 1000:
			return cnt * bscore
		b = (b - 1 + next(die) + next(die) + next(die)) % 10 + 1
		bscore += b
		cnt += 3
		if bscore >= 1000:
			return cnt * ascore


def simulate(init, moves, possibleturns):
	agenda = [(init, 0, 1, 1)]
	winsat = [0] * 20
	nowinsat = [0] * 20
	while agenda:
		pos, score, nummoves, mult = agenda.pop()
		for turn, cnt in possibleturns.items():
			steps = moves[pos, turn]
			newscore = score + steps
			if newscore >= 21:
				winsat[nummoves] += cnt * mult
			else:
				nowinsat[nummoves] += cnt * mult
				agenda.append((steps, newscore, nummoves + 1, cnt * mult))
	return winsat, nowinsat


def day21b(s):
	_, a, _, b = map(int, re.findall(r'\d+', s))
	possibleturns = Counter(sum(a)
			for a in itertools.product([1, 2, 3], repeat=3))
	moves = {}  # (startpos, steps) -> score increase
	for init in range(1, 11):
		for turn in possibleturns:
			moves[init, turn] = (init - 1 + turn) % 10 + 1
	awinsat, anowinsat = simulate(a, moves, possibleturns)
	bwinsat, bnowinsat = simulate(b, moves, possibleturns)
	awins = sum(awinsat[n] * bnowinsat[n - 1] for n in range(1, 20))
	bwins = sum(bwinsat[n] * anowinsat[n] for n in range(1, 20))
	return max(awins, bwins)


def day22a(s):
	grid = np.zeros((101, 101, 101), dtype=bool)
	prev = 0
	for l in s.splitlines():
		bit = l.startswith('on')
		xx, yy, zz = l.split()[1].split(',')
		x1, x2 = map(int, xx.split('=')[1].split('..'))
		y1, y2 = map(int, yy.split('=')[1].split('..'))
		z1, z2 = map(int, zz.split('=')[1].split('..'))
		if any(c < -50 or c > 50 for c in (x1, x2, y1, y2, z1, z2)):
			continue
		grid[x1 + 50:x2 + 51, y1 + 50:y2 + 51, z1 + 50:z2 + 51] = bit
		print(grid.sum() - prev, grid.sum(), l)
		prev = grid.sum()
	return grid.sum()


def volume(x1, x2, y1, y2, z1, z2):
	return (x2 - x1) * (y2 - y1) * (z2 - z1)


def overlap(x1, x2, y1, y2, z1, z2, _, u1, u2, v1, v2, w1, w2):
	x1 = x1 if x1 > u1 else u1
	x2 = x2 if x2 < u2 else u2
	y1 = y1 if y1 > v1 else v1
	y2 = y2 if y2 < v2 else v2
	z1 = z1 if z1 > w1 else w1
	z2 = z2 if z2 < w2 else w2
	if x1 < x2 and y1 < y2 and z1 < z2:
		return x1, x2, y1, y2, z1, z2


def day22b(s):
	steps = []
	for l in s.splitlines():
		bit = l.startswith('on')
		xx, yy, zz = l.split()[1].split(',')
		x1, x2 = sorted(map(int, xx.split('=')[1].split('..')))
		y1, y2 = sorted(map(int, yy.split('=')[1].split('..')))
		z1, z2 = sorted(map(int, zz.split('=')[1].split('..')))
		step = bit, x1, x2 + 1, y1, y2 + 1, z1, z2 + 1
		steps.append(step)
	deltas = Counter()
	for step in steps:
		update = Counter()
		for prev, cnt in deltas.items():
			if cnt:
				coords = overlap(*prev, *step)
				if coords:
					update[coords] -= cnt
		update[step[1:]] += step[0]
		deltas.update(update)
	return sum(volume(*step) * cnt for step, cnt in deltas.items())


if __name__ == '__main__':
	main(globals())
