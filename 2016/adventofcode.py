"""http://adventofcode.com/2016 """
import re
import sys
# import operator
import itertools
# from functools import reduce
from collections import Counter, defaultdict
from heapq import heappush, heappop
# import numpy as np
# from numba import njit
sys.path.append('..')
from common import main


def day1(s):
	"""
	>>> day1('R2, L3')
	(5, -1)
	>>> day1('R2, R2, R2')
	(2, -1)
	>>> day1('R5, L5, R5, R3')
	(12, -1)
	>>> day1('R8, R4, R4, R8')
	(8, 4)
	"""
	x = y = dir = 0
	ans2 = -1
	seen = [(x, y)]
	for a, b in re.findall(r'(L|R)(\d+)', s):
		dist = int(b)
		dir = ((dir - 1) % 4) if a == 'L' else ((dir + 1) % 4)
		if dir == 0:
			for yy in range(y + 1, y + dist + 1):
				if ans2 == -1 and (x, yy) in seen:
					ans2 = abs(x) + abs(yy)
				seen.append((x, yy))
			y += dist
		elif dir == 1:
			for xx in range(x + 1, x + dist + 1):
				if ans2 == -1 and (xx, y) in seen:
					ans2 = abs(xx) + abs(y)
				seen.append((xx, y))
			x += dist
		elif dir == 2:
			for yy in range(y - 1, y - dist - 1, -1):
				if ans2 == -1 and (x, yy) in seen:
					ans2 = abs(x) + abs(yy)
				seen.append((x, yy))
			y -= dist
		elif dir == 3:
			for xx in range(x - 1, x - dist - 1, -1):
				if ans2 == -1 and (xx, y) in seen:
					ans2 = abs(xx) + abs(y)
				seen.append((xx, y))
			x -= dist
	ans1 = abs(x) + abs(y)
	return ans1, ans2


def day2(s):
	def _day2(s, buttons, x, y):
		code = ''
		for line in s.splitlines():
			for a in line:
				if a == 'U' and y and buttons[y - 1][x]:
					y -= 1
				elif a == 'D' and y + 1 < len(buttons) and buttons[y + 1][x]:
					y += 1
				elif a == 'L' and x and buttons[y][x - 1]:
					x -= 1
				elif a == 'R' and x + 1 < len(buttons[0]) and buttons[y][x + 1]:
					x += 1
			code += str(buttons[y][x])
		return code

	buttons1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	buttons2 = [
			[0, 0, 1, 0, 0],
			[0, 2, 3, 4, 0],
			[5, 6, 7, 8, 9],
			[0, 'A', 'B', 'C', 0],
			[0, 0, 'D', 0, 0]]
	return _day2(s, buttons1, 1, 1), _day2(s, buttons2, 0, 2)


def day3(s):
	def possibletriangle(a, b, c):
		return a + b > c and a + c > b and b + c > a

	data = [[int(a) for a in line.split()]
			for line in s.splitlines()]
	count = sum(possibletriangle(a, b, c) for a, b, c in data)
	count2 = 0
	for n in range(0, len(data), 3):
		for m in range(3):
			a, b, c = [data[n + nn][m] for nn in range(3)]
			count2 += possibletriangle(a, b, c)
	return count, count2


def day4(s):
	result = 0
	pattern = re.compile(r'([-a-z]+)-([0-9]+)\[([a-z]+)\]')
	for line in s.splitlines():
		room, sector, checksum = pattern.match(line).groups()
		room = room.replace('-', '')
		cnt = Counter(sorted(room))
		if checksum == ''.join(a for a, _ in cnt.most_common(5)):
			result += int(sector)
	for line in s.splitlines():
		room, sector, checksum = pattern.match(line).groups()
		sector = int(sector)
		room = ''.join(' ' if a == '-' else
				chr(((ord(a) - 96) + sector) % 26 + 96)
				for a in room)
		if room == 'northpole object storage':
			return result, sector


def _day5(s):
	try:
		from _md5 import md5  # https://stackoverflow.com/a/60263898/338811
	except ImportError:
		from hashlib import md5
	hasher = md5()
	hasher.update(s.encode('ascii'))
	m = 0
	while True:
		x = ''
		while not x.startswith('00000'):
			newhash = hasher.copy()
			newhash.update(b'%d' % m)
			x = newhash.hexdigest()
			m += 1
		yield x


def day5a(s):
	return ''.join(a[5] for a in itertools.islice(_day5(s), 8))


def day5b(s):
	result = [None] * 8
	hashes = iter(_day5(s))
	while None in result:
		x = next(hashes)
		pos = int(x[5], base=16)
		if pos < 8 and result[pos] is None:
			result[pos] = x[6]
	return ''.join(result)


def day6(s):
	result = result2 = ''
	lines = s.splitlines()
	for n in range(len(lines[0])):
		cnt = Counter(line[n] for line in lines).most_common()
		result += cnt[0][0]
		result2 += cnt[-1][0]
	return result, result2


def day7a(s):
	def hasabba(part):
		return any(a != b for a, b in abba.findall(part))

	hyper = re.compile(r'\[(.*?)\]')
	abba = re.compile(r'(.)(.)\2\1')
	count = 0
	for line in s.splitlines():
		if (not any(hasabba(part) for part in hyper.findall(line))
				and hasabba(line)):
			count += 1
	return count


def day7b(s):
	hyper = re.compile(r'\[(.*?)\]')
	count = 0
	for line in s.splitlines():
		x = hyper.sub('  ', line)
		y = hyper.findall(line)
		if any(a == c and a != b
				and any(b + a + b in part for part in y)
				for a, b, c in zip(x, x[1:], x[2:])):
			count += 1
	return count


def day8(s):
	# rowlen, numrows = 7, 3
	rowlen, numrows = 50, 6
	r = [[' ' for _ in range(rowlen)] for _ in range(numrows)]
	for line in s.splitlines():
		if line.startswith('rect'):
			_, ab = line.split()
			a, b = ab.split('x')
			for n in range(int(a)):
				for m in range(int(b)):
					r[m][n] = '#'
		elif line.startswith('rotate row'):
			_, _, a, _, b = line.split()
			y, b = int(a.split('=')[1]), int(b)
			r[y] = [r[y][(n - b) % rowlen] for n in range(rowlen)]
		elif line.startswith('rotate column'):
			_, _, a, _, b = line.split()
			x, b = int(a.split('=')[1]), int(b)
			newcol = [r[(y - b) % numrows][x] for y in range(numrows)]
			for y in range(numrows):
				r[y][x] = newcol[y]
	result = '\n'.join(''.join(line) for line in r)
	return '%s\n%s' % (result.count('#'), result)


def day9(s):
	"""
	>>> day9('ADVENT')
	6
	>>> day9('A(1x5)BC')
	7
	>>> day9('(3x3)XYZ')
	9
	>>> day9('A(2x2)BCD(2x2)EFG')
	11
	>>> day9('(6x1)(1x3)A')
	6
	>>> day9('X(8x2)(3x3)ABCY')
	18
	"""
	result = ''
	n = 0
	while n < len(s):
		if s[n] == '(':
			end = s.index(')', n)
			a, b = s[n + 1:end].split('x')
			a, b = int(a), int(b)
			result += s[end + 1:end + a + 1] * b
			n = end + a + 1
		else:
			result += s[n]
			n += 1
	return len(result)


def day9b(s):
	"""
	>>> day9b('(3x3)XYZ')
	9
	>>> day9b('X(8x2)(3x3)ABCY')
	20
	>>> day9b('(27x12)(20x12)(13x14)(7x10)(1x12)A')
	241920
	>>> day9b('(25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN')
	445
	"""
	def decompress(s):
		result = 0
		n = 0
		while n < len(s):
			if s[n] == '(':
				end = s.index(')', n)
				a, b = s[n + 1:end].split('x')
				a, b = int(a), int(b)
				data = s[end + 1:end + a + 1]
				if '(' in data:
					result += decompress(data) * b
				else:
					result += a * b
				n = end + a + 1
			else:
				result += 1
				n += 1
		return result

	return decompress(s)


def day10(s):
	state = defaultdict(list)
	outputs = {}
	lowhigh = {}
	for line in s.splitlines():
		x = line.split()
		if x[0] == 'value':
			state[int(x[-1])].append(int(x[1]))
		elif x[0] == 'bot':
			lowhigh[int(x[1])] = (x[5], int(x[6])), (x[-2], int(x[-1]))
		else:
			raise ValueError
	while state:
		for bot, chips in [(a, b) for a, b in state.items() if len(b) == 2]:
			if 61 in chips and 17 in chips:
				result1 = bot
			if bot in lowhigh:
				low, high = lowhigh[bot]
				if low[0] == 'output':
					outputs[low[1]] = min(chips)
				else:
					state[low[1]].append(min(chips))
				if high[0] == 'output':
					outputs[high[1]] = max(chips)
				else:
					state[high[1]].append(max(chips))
				state.pop(bot)
			else:
				raise ValueError
	return result1, outputs[0] * outputs[1] * outputs[2]


def day11a(s):
	def srepr(elevator, state):
		return hash(tuple([elevator] + sorted(zip(state, state[1:]))))

	def estimate(elevator, state):
		return sum([5, 3, 1, 0][b] for b in state) + (3 - elevator)

	def legal(state):
		return not any(
				(a & 1)  # is a microchip
				and b != state[a - 1]  # not on same floor as its generator
				and any((c & 1) == 0 and b == d  # microchip with diff gen.
						for c, d in enumerate(state))
				for a, b in enumerate(state))

	state = {a: n for n, line in enumerate(s.splitlines())
			for a in re.findall(
				r'(\w+(?: generator|-compatible microchip))', line)}
	state = [b for _, b in sorted(state.items())]
	agenda = [[(0, 0, state)]] + [[] for _ in range(100)]
	seen = {srepr(0, state)}
	curmin = 0
	while True:
		while not agenda[curmin]:
			curmin += 1
		steps, elevator, state = agenda[curmin].pop()
		if all(b == 3 for b in state):
			return steps
		onfloor = [a for a, b in enumerate(state) if b == elevator]
		for comb in itertools.chain(
				((a, ) for a in onfloor),
				itertools.combinations(onfloor, 2)):
			for dir in (1, -1):
				if 0 <= elevator + dir <= 3:
					newstate = [b + dir * (a in comb)
							for a, b in enumerate(state)]
					if legal(newstate) and srepr(
							elevator + dir, newstate) not in seen:
						seen.add(srepr(elevator + dir, newstate))
						est = estimate(elevator, newstate) + steps + 1
						agenda[est].append((
								steps + 1, elevator + dir, newstate))
						if curmin > est:
							curmin = est


def day11b(s):
	lines = s.splitlines()
	lines[0] += ' elerium generator elerium-compatible microchip'
	lines[0] += ' dilithium generator dilithium-compatible microchip'
	return day11a('\n'.join(lines))


def day12a(s):
	reg = dict.fromkeys('abcd', 0)
	return day12(s, reg)


def day12b(s):
	reg = dict.fromkeys('abcd', 0)
	reg['c'] = 1
	return day12(s, reg)


def day12(s, reg):
	def decode(a):
		return int(a) if a.isdigit() else reg[a]

	pc = 0
	lines = [line.split() for line in s.splitlines()]
	while pc < len(lines):
		if lines[pc][0] == 'cpy':
			reg[lines[pc][2]] = decode(lines[pc][1])
		elif lines[pc][0] == 'inc':
			reg[lines[pc][1]] += 1
		elif lines[pc][0] == 'dec':
			reg[lines[pc][1]] -= 1
		elif lines[pc][0] == 'jnz':
			if decode(lines[pc][1]) != 0:
				pc += int(lines[pc][2])
				continue
		else:
			raise ValueError(lines[pc])
		pc += 1
	return reg['a']


def day13a(s):
	def heur(steps, x, y):
		return abs(goalx - x) + abs(goaly - y), steps, x, y

	num, goalx, goaly = [int(a) for a in s.split()]
	grid = {}  # np.zeros((2 * goalx, 2 * goaly))
	for x in range(2 * goalx):
		for y in range(2 * goaly):
			grid[x, y] = 0
			val = x * x + 3 * x + 2 * x * y + + y + y * y
			if bin(val + num).count('1') & 1:
				grid[x, y] = 1
	agenda = [(goalx + goaly, 0, 1, 1)]
	while agenda:
		_, steps, x, y = heappop(agenda)
		grid[x, y] = 1  # mark as visited to prevent cycles
		if x == goalx and y == goaly:
			break
		if grid.get((x + 1, y), 0) == 0:
			heappush(agenda, heur(steps + 1, x + 1, y))
		if grid.get((x - 1, y), 0) == 0 and x > 0:
			heappush(agenda, heur(steps + 1, x - 1, y))
		if grid.get((x, y + 1), 0) == 0:
			heappush(agenda, heur(steps + 1, x, y + 1))
		if grid.get((x, y - 1), 0) == 0 and y > 0:
			heappush(agenda, heur(steps + 1, x, y - 1))
	return steps


def day13b(s):
	def heur(steps, x, y):
		return steps, x, y

	num, goalx, goaly = [int(a) for a in s.split()]
	grid = {}  # np.zeros((2 * goalx, 2 * goaly))
	for x in range(2 * goalx):
		for y in range(2 * goaly):
			grid[x, y] = 0
			val = x * x + 3 * x + 2 * x * y + + y + y * y
			if bin(val + num).count('1') & 1:
				grid[x, y] = 1
	agenda = [(0, 1, 1)]
	while agenda:
		steps, x, y = heappop(agenda)
		if steps > 50:
			break
		grid[x, y] = 2  # mark as visited to prevent cycles
		if grid.get((x + 1, y), 0) == 0:
			heappush(agenda, heur(steps + 1, x + 1, y))
		if grid.get((x - 1, y), 0) == 0 and x > 0:
			heappush(agenda, heur(steps + 1, x - 1, y))
		if grid.get((x, y + 1), 0) == 0:
			heappush(agenda, heur(steps + 1, x, y + 1))
		if grid.get((x, y - 1), 0) == 0 and y > 0:
			heappush(agenda, heur(steps + 1, x, y - 1))
	return sum(a == 2 for a in grid.values())


def day14a(s):
	return day14(s, iterations=0)


def day14b(s):
	return day14(s, iterations=2016)


def day14(s, iterations=0):
	def get(n):
		if n in cache:
			x = cache[n]
		else:
			newhash = hasher.copy()
			newhash.update(b'%d' % n)
			x = newhash.hexdigest()
			for _ in range(iterations):
				x = md5(x.encode('ascii')).hexdigest()
			cache[n] = x
		return x

	try:
		from _md5 import md5  # https://stackoverflow.com/a/60263898/338811
	except ImportError:
		from hashlib import md5
	hasher = md5()
	hasher.update(s.encode('ascii'))
	triple = re.compile(r'(.)\1\1')
	cache = {}
	n = 0
	foundkeys = 0
	while True:
		match = triple.search(get(n))
		if match is not None:
			quintuple = re.compile(r'%s' % (5 * match.group(1)))
			for m in range(n + 1, n + 1001):
				if quintuple.search(get(m)) is not None:
					foundkeys += 1
					print(foundkeys)
					if foundkeys >= 64:
						return n
					break
		n += 1


def day15(s):
	def solve(discs):
		for n in range(
				discs[0][1] - discs[0][3] - discs[0][0],
				1000000000000,
				discs[0][1]):
			if all((n + d + pos) % ppos == 0 for d, ppos, _, pos in discs):
				return n

	discs = [list(map(int, re.findall(r'\d+', a))) for a in s.splitlines()]
	return solve(discs), solve(discs + [[7, 11, 0, 0]])


def day16(s):
	def solve(goal):
		data = s
		while len(data) < goal:
			b = data[::-1].replace('1', '2').replace('0', '1').replace('2', '0')
			data += '0' + b
		data = data[:goal]
		checksum = ''
		while True:
			for a, b in zip(data[::2], data[1::2]):
				if a == b:
					checksum += '1'
				else:
					checksum += '0'
			if (len(checksum) & 1) != 0:
				break
			print(checksum, len(checksum))
			data = checksum
			checksum = ''
		return checksum

	return solve(272), solve(35651584)


if __name__ == '__main__':
	main(globals())
