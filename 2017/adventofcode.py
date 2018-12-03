"""Advent of Code 2017. http://adventofcode.com/2017 """
import sys
import array
from collections import Counter, defaultdict
from operator import xor
from functools import reduce
from itertools import count
from binascii import hexlify
import numpy as np


def day1a(s):
	"""http://adventofcode.com/2017/day/1"""
	return sum(int(a) for a, b in zip(s, s[1:] + s[0]) if a == b)


def day1b(s):
	return sum(int(a) for n, a in enumerate(s, len(s) // 2)
			if s[n % len(s)] == a)


def day2a(s):
	"""http://adventofcode.com/2017/day/2"""
	data = [[int(a) for a in line.split()]
			for line in s.splitlines()]
	return sum(max(row) - min(row) for row in data)


def day2b(s):
	data = [[int(a) for a in line.split()]
			for line in s.splitlines()]
	return sum(a // b for row in data
			for a in row
				for b in row
					if a > b and a % b == 0)


def day3a(s):
	"""http://adventofcode.com/2017/day/3"""
	s = int(s)
	n = steps = 1
	x = y = 0
	while True:
		if n == s:
			return abs(x) + abs(y)
		for _ in range(steps):
			n += 1
			x += 1
			if n == s:
				return abs(x) + abs(y)
		for _ in range(steps):
			n += 1
			y += 1
			if n == s:
				return abs(x) + abs(y)
		steps += 1
		for _ in range(steps):
			n += 1
			x -= 1
			if n == s:
				return abs(x) + abs(y)
		for _ in range(steps):
			n += 1
			y -= 1
			if n == s:
				return abs(x) + abs(y)
		steps += 1


def day3b(s):
	def neighbors(x, y):
		return (
			table.get((x - 1, y), 0)
			+ table.get((x - 1, y + 1), 0)
			+ table.get((x - 1, y - 1), 0)
			+ table.get((x + 1, y), 0)
			+ table.get((x + 1, y + 1), 0)
			+ table.get((x + 1, y - 1), 0)
			+ table.get((x, y + 1), 0)
			+ table.get((x, y - 1), 0))

	s = int(s)
	steps = 1
	x = y = 0
	table = {(0, 0): 1}
	while True:
		if table[x, y] > s:
			return table[x, y]
		for _ in range(steps):
			x += 1
			table[x, y] = neighbors(x, y)
			if table[x, y] > s:
				return table[x, y]
		for _ in range(steps):
			y += 1
			table[x, y] = neighbors(x, y)
			if table[x, y] > s:
				return table[x, y]
		steps += 1
		for _ in range(steps):
			x -= 1
			table[x, y] = neighbors(x, y)
			if table[x, y] > s:
				return table[x, y]
		for _ in range(steps):
			y -= 1
			table[x, y] = neighbors(x, y)
			if table[x, y] > s:
				return table[x, y]
		steps += 1


def day4a(s):
	"""http://adventofcode.com/2017/day/4"""
	return sum(max(Counter(line.split()).values()) == 1
			for line in s.splitlines())


def day4b(s):
	return sum(max(Counter(
			''.join(sorted(word)) for word in line.split()).values()) == 1
			for line in s.splitlines())


def day5a(s):
	"""http://adventofcode.com/2017/day/5"""
	maze = [int(a) for a in s.splitlines()]
	n = pos = 0
	while 0 <= pos < len(maze):
		newpos = pos + maze[pos]
		maze[pos] += 1
		pos = newpos
		n += 1
	return n


def day5b(s):
	maze = [int(a) for a in s.splitlines()]
	n = pos = 0
	while 0 <= pos < len(maze):
		newpos = pos + maze[pos]
		maze[pos] += -1 if maze[pos] >= 3 else 1
		pos = newpos
		n += 1
	return n


def day6a(s):
	"""http://adventofcode.com/2017/day/6"""
	data = np.array([int(a) for a in s.split()], dtype=np.int8)
	lendata1 = len(data) - 1
	seen = set()
	n = 0
	while data.tobytes() not in seen:
		seen.add(data.tobytes())
		pos = data.argmax()
		blocks = data[pos]
		data[pos] = 0
		for x in range(pos + 1, blocks + pos + 1):
			data[x & lendata1] += 1  # x % len(data)
		n += 1
	return n


def day6b(s):
	data = np.array([int(a) for a in s.split()], dtype=np.int8)
	lendata1 = len(data) - 1
	seen = set()
	n = 0
	while data.tobytes() not in seen:
		seen.add(data.tobytes())
		pos = data.argmax()
		blocks = data[pos]
		data[pos] = 0
		for x in range(pos + 1, blocks + pos + 1):
			data[x & lendata1] += 1  # x % len(data)
		n += 1
	origdata = data.copy()
	n = 0
	while True:
		pos = data.argmax()
		blocks = data[pos]
		data[pos] = 0
		for x in range(pos + 1, blocks + pos + 1):
			data[x & lendata1] += 1  # x % len(data)
		n += 1
		if np.array_equal(origdata, data):
			return n


def day7a(s):
	"""http://adventofcode.com/2017/day/7"""
	parent = {}
	for line in s.splitlines():
		node = line.split(' (')[0]
		if '->' in line:
			children = line.split(' -> ')[1].split(', ')
			for child in children:
				parent[child] = node
	node = next(iter(parent))
	while node in parent:
		node = parent[node]
	return node


def day7b(s):
	parent = {}
	children = {}
	weight = {}
	for line in s.splitlines():
		node, x = line.split(' (', 1)
		weight[node] = int(x.split(')', 1)[0])
		if '->' in line:
			children1 = line.split(' -> ')[1].split(', ')
			children[node] = children1
			for child in children1:
				parent[child] = node
		else:
			children[node] = []
	node = next(iter(parent))
	while node in parent:
		node = parent[node]
	root = node

	def totalweight(node):
		return weight[node] + sum(
				totalweight(child) for child in children[node])

	def postorder(node):
		for child in children[node]:
			yield from postorder(child)
		yield node

	for node in postorder(root):
		w = [totalweight(child) for child in children[node]]
		cw = Counter(w)
		if len(cw) > 1:
			lcw = [a for a, _ in cw.most_common()]
			a, b = lcw
			bc = children[node][w.index(b)]
			return weight[bc] - (b - a)


def day8a(s):
	"""http://adventofcode.com/2017/day/8"""
	registers = defaultdict(int)
	for line in s.splitlines():
		instruction = line.split()
		reg = instruction[0]
		op = instruction[1]
		val = int(instruction[2])
		assert instruction[3] == 'if'
		condval1 = registers[instruction[4]]
		condval2 = int(instruction[6])
		condop = instruction[5]
		cond = eval('%d %s %d' % (condval1, condop, condval2))
		if cond:
			if op == 'inc':
				registers[reg] += val
			elif op == 'dec':
				registers[reg] -= val
			else:
				raise ValueError
	return max(registers.values())


def day8b(s):
	curmax = -1
	registers = defaultdict(int)
	for line in s.splitlines():
		instruction = line.split()
		reg = instruction[0]
		op = instruction[1]
		val = int(instruction[2])
		assert instruction[3] == 'if'
		condval1 = registers[instruction[4]]
		condval2 = int(instruction[6])
		condop = instruction[5]
		cond = eval('%d %s %d' % (condval1, condop, condval2))
		if cond:
			if op == 'inc':
				registers[reg] += val
			elif op == 'dec':
				registers[reg] -= val
			else:
				raise ValueError
			curmax = max(curmax, max(registers.values()))
	return curmax


def day9a(s):
	"""http://adventofcode.com/2017/day/9"""
	pos = cur = result = 0
	while pos < len(s):
		if s[pos] == '{':
			cur += 1
			result += cur
			pos += 1
		elif s[pos] == '}':
			cur -= 1
			pos += 1
		elif s[pos] == '<':
			while s[pos] != '>':
				if s[pos] == '!':
					pos += 1
				pos += 1
			pos += 1
			continue
		elif s[pos] == ',':
			pos += 1
		else:
			raise ValueError('%d %s' % (pos, s[pos]))
	return result


def day9b(s):
	pos = cur = result = 0
	while pos < len(s):
		if s[pos] == '{':
			cur += 1
			pos += 1
		elif s[pos] == '}':
			cur -= 1
			pos += 1
		elif s[pos] == '<':
			while s[pos] != '>':
				if s[pos] == '!':
					pos += 2
				else:
					pos += 1
					result += 1
			result -= 1
			pos += 1
			continue
		elif s[pos] == ',':
			pos += 1
		else:
			raise ValueError('%d %s' % (pos, s[pos]))
	return result


def day10a(s, elements=256):
	"""http://adventofcode.com/2017/day/10"""
	lens = np.array([int(a) for a in s.split(',')])
	lst = np.arange(elements)
	pos = skip = 0
	for l in lens:
		lst[np.arange(pos, pos + l) % elements] = lst[
				np.arange(pos, pos + l) % elements][::-1]
		pos += l + skip
		pos = pos % elements
		skip += 1
	return lst[0] * lst[1]


def day10b(s, elements=256):
	lens = [ord(a) for a in s] + [17, 31, 73, 47, 23]
	lst = np.arange(elements)
	pos = skip = 0
	elements1 = elements - 1
	for n in range(64):
		for l in lens:
			indices = np.arange(pos, pos + l) & elements1
			lst[indices] = lst[indices][::-1]
			pos = (pos + l + skip) & elements1
			skip += 1
	return hexlify(bytes(reduce(xor, lst[n:n + 16]) for n in range(0, 256, 16))
			).decode('ascii')


def day11(s):
	"""http://adventofcode.com/2017/day/11"""
	path = s.split(',')
	maxd = x = y = z = 0
	for a in path:
		if a == 'n':
			x += 1
			y -= 1
		elif a == 's':
			x -= 1
			y += 1
		elif a == 'ne':
			y -= 1
			z += 1
		elif a == 'sw':
			y += 1
			z -= 1
		elif a == 'se':
			x -= 1
			z += 1
		elif a == 'nw':
			x += 1
			z -= 1
		maxd = max(maxd, max(abs(x), abs(y), abs(z)))
	return max(abs(x), abs(y), abs(z)), maxd


def day11a(s):
	return day11(s)[0]


def day11b(s):
	return day11(s)[1]


def day12a(s):
	"""http://adventofcode.com/2017/day/12"""
	graph = {}
	for line in s.splitlines():
		a, b = line.split(' <-> ')
		graph[int(a)] = [int(c) for c in b.split(', ')]
	reachable = 0
	queue = [0]
	visited = set()
	while queue:
		node = queue.pop(0)
		if node not in visited:
			reachable += 1
			queue.extend(graph.get(node, []))
		visited.add(node)
	return reachable


def day12b(s):
	graph = {}
	for line in s.splitlines():
		src, targets = line.split(' <-> ')
		graph[int(src)] = [int(target) for target in targets.split(', ')]
	candidates = list(graph)
	visited = set()
	groups = {}
	while candidates:
		a = candidates.pop()
		if a in visited:
			continue
		groups[a] = []
		queue = [a]
		while queue:
			node = queue.pop(0)
			if node not in visited:
				groups[a].append(node)
				queue.extend(graph.get(node, []))
			visited.add(node)
	return len(groups)


def day13a(s):
	"""http://adventofcode.com/2017/day/13"""
	depths = []
	ranges = []
	for line in s.splitlines():
		a, b = line.split(': ')
		depths.append(int(a))
		ranges.append(int(b))
	position = -1
	severity = 0
	down = -1
	up = 1
	scanners = [0 for _ in depths]
	direction = [up for _ in depths]
	while position < max(depths):
		position += 1
		if position in depths:
			n = depths.index(position)
			if scanners[n] == 0:
				severity += depths[n] * ranges[n]
		for n, a in enumerate(ranges):
			if scanners[n] == 0:
				direction[n] = up
			elif scanners[n] + 1 == ranges[n]:
				direction[n] = down
			scanners[n] += direction[n]
	return severity


def day13b(s):
	ranges = []
	for line in s.splitlines():
		a, b = line.split(': ')
		ranges.extend([None] * (int(a) - len(ranges)))
		ranges.append(int(b))

	for delay in count():
		time = delay
		for scanner in ranges:
			if (scanner is not None
					and time % (scanner * 2 - 2) == 0):
				break  # caught
			time += 1
		else:  # not caught
			return delay


def day14a(s):
	"""http://adventofcode.com/2017/day/14"""
	return sum(bin(int(day10b('%s-%d' % (s, n)), 16)).count('1')
			for n in range(128))


def day14b(s):
	def floodfill(buf, x, y, target, repl):
		if (target == repl
				or x < 0 or y < 0
				or x >= len(buf) or y >= len(buf)
				or buf[x, y] != target):
			return
		buf[x, y] = repl
		floodfill(buf, x - 1, y, target, repl)
		floodfill(buf, x + 1, y, target, repl)
		floodfill(buf, x, y - 1, target, repl)
		floodfill(buf, x, y + 1, target, repl)

	buf = np.array([
			list(map(int, format(int(day10b('%s-%d' % (s, n)), 16), '0128b')))
			for n in range(128)])
	n = 1
	for x in range(128):
		for y in range(128):
			if buf[x, y] == 1:
				n += 1
				floodfill(buf, x, y, 1, n)
	return n - 1


def day15a(s, cycles=int(40e6)):
	"""http://adventofcode.com/2017/day/15"""
	a, b = s.splitlines()
	a, b = int(a.split()[-1]), int(b.split()[-1])
	result = 0
	for n in range(cycles):
		a *= 16807
		b *= 48271
		a %= 2147483647
		b %= 2147483647
		result += (a & 0xffff) == (b & 0xffff)
	return result


def day15b(s, cycles=int(5e6)):
	def gen(start, factor, mask):
		a = start
		while True:
			a = (a * factor) % 2147483647
			if (a & mask) == 0:
				yield a

	a, b = s.splitlines()
	a, b = int(a.split()[-1]), int(b.split()[-1])
	return sum(
			(a & 0xffff) == (b & 0xffff)
			for _, a, b in zip(
				range(cycles),
				gen(a, 16807, 3),
				gen(b, 48271, 7)))


def day16(s, letters):
	"""http://adventofcode.com/2017/day/16"""
	for move in s.strip().split(','):
		if move[0] == 's':
			i = int(move[1:])
			letters = letters[-i:] + letters[:-i]
		elif move[0] == 'x':
			i, j = move[1:].split('/')
			i, j = int(i), int(j)
			letters[i], letters[j] = letters[j], letters[i]
		elif move[0] == 'p':
			i, j = move[1:].split('/')
			i, j = letters.index(i), letters.index(j)
			letters[i], letters[j] = letters[j], letters[i]
	return letters


def day16a(s, numletters=16):
	letters = 'abcdefghijklmnop'[:numletters]
	return ''.join(day16(s, list(letters)))


def day16b(s, numletters=16, iterations=1000000000):
	letters = 'abcdefghijklmnop'[:numletters]
	firstidx = {letters: 0}
	results = {0: letters}
	cache = {}
	n = 1
	letters = ''.join(day16(s, list(letters)))
	firstidx[letters] = n
	results[n] = letters
	while True:
		if letters in firstidx and firstidx.get(letters, -1) != n:
			i = firstidx[letters]
			steps = n - i
			# FIXME: figure out how to make smaller steps without
			# resorting to incremental steps
			if n + steps > iterations:
				break
			letters = results[n]
			n += steps
		else:
			if n + 1 > iterations:
				break
			start = letters
			letters = ''.join(day16(s, list(letters)))
			cache[start] = letters
			n += 1
			if letters not in firstidx:
				firstidx[letters] = n
		results[n] = letters
	for m in range(n, iterations):
		if letters in cache:
			letters = cache[letters]
		else:
			start = letters
			letters = ''.join(day16(s, list(letters)))
			cache[start] = letters
	return ''.join(letters)


def day17a(s):
	"""http://adventofcode.com/2017/day/17"""
	steps = int(s)
	iterations = 2017
	target = 2017
	pos = val = 0
	buf = array.array('I', [0])
	# from collections import deque
	# buf = deque([0])
	for n in range(iterations):
		pos = ((pos + steps) % len(buf)) + 1
		val += 1
		buf.insert(pos, val)
	return buf[buf.index(target) + 1]


def day17b(s):
	steps = int(s)
	iterations = 50000000
	pos = 0
	numbefore = 0  # no of items after 0
	numafter = 0  # no of items after 0
	nextnum = None  # the number after 0
	for val in range(1, iterations + 1):
		pos = ((pos + steps) % (numbefore + 1 + numafter)) + 1
		if pos <= numbefore:
			numbefore += 1
		else:
			numafter += 1
			if pos == numbefore + 1:
				nextnum = val
	return nextnum


def day18a(s):
	"""http://adventofcode.com/2017/day/18"""
	program = s.splitlines()
	reg = {}
	pos = 0
	freq = None
	while True:
		if pos < 0 or pos >= len(program):
			break
		x = program[pos].split()
		op, op1 = x[0], x[1]
		if len(x) > 2:
			op2 = reg.get(x[2], 0) if 'a' <= x[2] <= 'z' else int(x[2])
		if op == 'snd':
			freq = reg.get(op1, 0)
		elif op == 'set':
			reg[op1] = int(op2)
		elif op == 'add':
			reg[op1] = reg.get(op1, 0) + int(op2)
		elif op == 'mul':
			reg[op1] = reg.get(op1, 0) * int(op2)
		elif op == 'mod':
			reg[op1] = reg.get(op1, 0) % int(op2)
		elif op == 'rcv' and reg[op1] != 0:
			break
		elif op == 'jgz' and reg[op1] > 0:
			pos += int(op2)
			continue
		pos += 1
	return freq


def day18b(s):
	def gen(program, pid):
		reg = {'p': pid}
		pos = 0
		while True:
			if pos < 0 or pos >= len(program):
				break
			x = program[pos].split()
			op, op1 = x[0], x[1]
			if len(x) > 2:
				op2 = reg.get(x[2], 0) if 'a' <= x[2] <= 'z' else int(x[2])
			if op == 'set':
				reg[op1] = int(op2)
			elif op == 'add':
				reg[op1] = reg.get(op1, 0) + int(op2)
			elif op == 'mul':
				reg[op1] = reg.get(op1, 0) * int(op2)
			elif op == 'mod':
				reg[op1] = reg.get(op1, 0) % int(op2)
			elif op == 'snd':
				yield 1, reg.get(op1, 0)
			elif op == 'rcv':
				reg[op1] = (yield 0, None)
			elif op == 'jgz' and (
					reg.get(op1, 0) > 0 if 'a' <= op1 <= 'z'
					else int(op1) > 0):
				pos += int(op2)
				continue
			pos += 1

	program = s.splitlines()
	x = [gen(program, 0), gen(program, 1)]
	queue0, queue1 = [], []
	state0 = state1 = 1  # 0=expecting data; 1=waiting to continue
	cnt = 0
	while True:
		if state0 == 1:
			state0, res = next(x[0])
			if state0 == 1:
				queue1.append(res)
		elif queue0 and state0 == 0:
			state0, res = x[0].send(queue0.pop(0))
			if state0 == 1:
				queue1.append(res)
		elif state1 == 1:
			state1, res = next(x[1])
			if state1 == 1:
				queue0.append(res)
				cnt += 1
		elif queue1 and state1 == 0:
			state1, res = x[1].send(queue1.pop(0))
			if state1 == 1:
				queue0.append(res)
				cnt += 1
		else:
			break
	return cnt


def day19(s):
	"""http://adventofcode.com/2017/day/19"""
	steps = 0
	diagram = s.splitlines()
	row = 0
	col = diagram[row].find('|')
	cur = diagram[row][col]
	direction = 'down'
	letters = []
	while cur != ' ':
		# change direction / collect letter
		if cur == '+':
			if direction in ('down', 'up'):
				if col > 0 and diagram[row][col - 1] not in (' ', '|'):
					direction = 'left'
				else:
					direction = 'right'
			elif direction in ('left', 'right'):
				if row > 0 and diagram[row - 1][col] not in (' ', '-'):
					direction = 'up'
				else:
					direction = 'down'
		elif cur != '|' and cur != '-':
			letters.append(cur)
		# move to next position
		if direction == 'down':
			row += 1
		elif direction == 'up':
			row -= 1
		elif direction == 'right':
			col += 1
		else:
			col -= 1
		cur = diagram[row][col]
		steps += 1
	return ''.join(letters), steps


def day19a(s):
	return day19(s)[0]


def day19b(s):
	return day19(s)[1]


def day20a(s, iterations=10000):
	"""http://adventofcode.com/2017/day/20"""
	p, v, a = [], [], []
	for line in s.splitlines():
		for x, y in zip(
				line.split(', '), (p, v, a)):
			y.append([int(z) for z in x.strip('pva=<>').split(',')])
	p, v, a = np.array(p), np.array(v), np.array(a)
	for n in range(iterations):
		# for x1, x2, x3 in zip(p, v, a):
		# 	print('%d. p=%s v=%s a=%s\t%d' % (n, x1, x2, x3, np.abs(x1).sum()))
		# print()
		v += a
		p += v
	return (np.abs(p).sum(axis=1).argmin(),
			np.abs(v).sum(axis=1).argmin(),
			np.abs(a).sum(axis=1).argmin())[1]


def day20b(s, iterations=10000):
	p, v, a = [], [], []
	for line in s.splitlines():
		for x, y in zip(
				line.split(', '), (p, v, a)):
			y.append([int(z) for z in x.strip('pva=<>').split(',')])
	p, v, a = np.array(p), np.array(v), np.array(a)
	for n in range(iterations):
		v += a
		p += v
		_, ind, cnt = np.unique(
				p, axis=0, return_index=True, return_counts=True)
		p = p[ind[cnt == 1]]
		v = v[ind[cnt == 1]]
		a = a[ind[cnt == 1]]
	return p.shape[0]


def day21(s, iterations):
	"""http://adventofcode.com/2017/day/21"""
	def parse(x):
		return np.array([list(map(int, a.replace('.', '0').replace('#', '1')))
					for a in x.split('/')], np.int8)

	def match(lhs, pat):
		for r in range(4):
			for f in range(4):
				lhs1 = np.flip(lhs, f) if f < 2 else lhs
				if f == 3:
					lhs1 = np.flip(np.flip(lhs, 0), 1)
				if np.array_equal(pat, np.rot90(lhs1, r)):
					return True
		return False

	buf = parse('.#./..#/###')
	rules = {2: [], 3: []}
	cache = {}
	for line in s.splitlines():
		lhs, rhs = line.split(' => ')
		rules[len(lhs.split('/'))].append((parse(lhs), parse(rhs)))
	for n in range(iterations):
		if (len(buf) & 1) == 0:
			window = 2
			newwindow = 3
			newsize = len(buf) // 2 * newwindow
		else:
			window = 3
			newwindow = 4
			newsize = len(buf) // 3 * newwindow
		newbuf = np.empty((newsize, newsize), np.int8)
		for a in range(0, len(buf), window):
			for b in range(0, len(buf), window):
				pat = buf[a:a + window, b:b + window]
				pat1 = tuple(pat.ravel())
				if pat1 in cache:
					aa = a // window * newwindow
					bb = b // window * newwindow
					newbuf[aa:aa + newwindow, bb:bb + newwindow] = cache[pat1]
				else:
					patones = pat.sum().sum()
					for lhs, rhs in rules[window]:
						if lhs.sum().sum() == patones:
							if match(lhs, pat):
								cache[pat1] = rhs
								aa = a // window * newwindow
								bb = b // window * newwindow
								newbuf[aa:aa + newwindow,
										bb:bb + newwindow] = rhs
								break
					else:
						print(n, a, b, window)
						raise ValueError('no matching rule.')
		buf = newbuf
	return buf.sum().sum()


def day21a(s, iterations=5):
	return day21(s, iterations)


def day21b(s, iterations=18):
	return day21(s, iterations)


def day22a(s, iterations=10000):
	"""http://adventofcode.com/2017/day/22"""
	gridsize = 10 if iterations <= 100 else 1000
	grid = np.zeros((gridsize, gridsize), np.int8)
	inp = np.array(
			[list(map(int, a.replace('.', '0').replace('#', '1')))
			for a in s.splitlines()], np.int8)
	grid[
			gridsize // 2 - len(inp) // 2:gridsize // 2 + len(inp) // 2 + 1,
			gridsize // 2 - len(inp) // 2:gridsize // 2 + len(inp) // 2 + 1,
			] = inp.T
	x = y = gridsize // 2
	direction = 0
	infected = 0
	for n in range(iterations):
		# if n < 10:
		# 	print('it', n)
		# 	for yy in range(gridsize):
		# 		for xx in range(gridsize):
		# 			if x == xx and y == yy:
		# 				print('[%s]' % ('#' if grid[xx, yy] else '.'), end='')
		# 			elif x == xx - 1 and y == yy:
		# 				print('%s' % ('#' if grid[xx, yy] else '.'), end='')
		# 			else:
		# 				print(' %s' % ('#' if grid[xx, yy] else '.'), end='')
		# 		print()
		if grid[x, y]:  # currently infected
			direction = (direction + 1) if direction < 3 else 0
			grid[x, y] = 0
		else:  # clean
			direction = (direction - 1) if direction else 3
			grid[x, y] = 1
			infected += 1
		# move forward
		if direction == 0:
			y -= 1
		elif direction == 1:
			x += 1
		elif direction == 2:
			y += 1
		elif direction == 3:
			x -= 1
	return infected


def day22b(s, iterations=10000000):
	gridsize = 10 if iterations <= 100 else 1024
	grid = np.zeros((gridsize, gridsize), np.int8)
	inp = np.array(
			[list(map(int, a.replace('.', '0').replace('#', '2')))
			for a in s.splitlines()], np.int8)
	x = y = gridsize // 2
	grid[
			x - len(inp) // 2:x + len(inp) // 2 + 1,
			y - len(inp) // 2:y + len(inp) // 2 + 1] = inp.T
	direction = 0  # up
	infected = 0
	# deltas for: up, right, down, left.
	dx = [0, 1, 0, -1]
	dy = [-1, 0, 1, 0]
	for n in range(iterations):
		# if n < 10:
		# 	print('it', n)
		# 	for yy in range(gridsize):
		# 		for xx in range(gridsize):
		# 			z = '.W#F'[grid[xx, yy]]
		# 			if x == xx and y == yy:
		# 				print('[%s]' % z, end='')
		# 			elif x == xx - 1 and y == yy:
		# 				print('%s' % z, end='')
		# 			else:
		# 				print(' %s' % z, end='')
		# 		print()
		direction = (direction + grid[x, y] - 1) & 3
		infected += (grid[x, y] == 1)
		grid[x, y] = (grid[x, y] + 1) & 3
		# move forward
		x += dx[direction]
		y += dy[direction]
	return infected


def day23a(s):
	"""http://adventofcode.com/2017/day/23"""
	program = s.splitlines()
	reg = dict.fromkeys('abcdefgh', 0)
	pos = mul = 0
	while 0 <= pos < len(program):
		x = program[pos].split()
		op, op1 = x[0], x[1]
		if len(x) > 2:
			op2 = reg.get(x[2], 0) if 'a' <= x[2] <= 'h' else int(x[2])
		if op == 'set':
			reg[op1] = int(op2)
		elif op == 'sub':
			reg[op1] = reg.get(op1, 0) - int(op2)
		elif op == 'mul':
			reg[op1] = reg.get(op1, 0) * int(op2)
			mul += 1
		elif op == 'jnz' and (
				reg.get(op1, 0) != 0
				if 'a' <= op1 <= 'h'
				else op1 != '0'):
			pos += int(op2)
			continue
		pos += 1
	return mul


def day23b(s):
	r"""Code translated to C and optimized by hand.
Compile and run: gcc -lm day23b.c && ./a.out
---------8<--------8<--------8<----------------------
#include <stdio.h>
#include <math.h>

int isprime(int n) {
	int sqrtn = sqrt(n);
	for (int m=2; m < sqrtn; m++) {
		if (n % m == 0)
			return 0;
	}
	return 1;
}

int main() {
	int a = 1, b = 0, c = 0, f = 0, h = 0;
	/* int d = 0, e = 0 */
	if (a == 0) {
		b = 84;
		c = 84;
	} else {
		b = 84 * 100 + 100000;
		c = b + 17000;
	}
	while (1) {
		/*
		f = 1;
		for (d = 2; d < b; d += 1) {
			for (e = 2; e < b; e += 1) {
				if (d * e == b)
					f = 0;
			}
		}
		*/
		f = isprime(b);
		if (f == 0)
			h += 1;
		if (b == c) {
			printf("%d\n", h);
			return 0;
		}
		b += 17;
	}
}"""
	def isprime(n):
		if n & 1 == 0:
			return False
		sqrtn = int(n ** 0.5)
		for m in range(3, sqrtn, 2):
			if n % m == 0:
				return False
		return True

	a = 1
	b = c = f = h = 0
	if a == 0:
		b = c = 84
	else:
		b = 84 * 100 + 100000
		c = b + 17000
	while 1:
		if not isprime(b):
			h += 1
		if b == c:
			return h
		b += 17


def day24(n, workingset):
	"""http://adventofcode.com/2017/day/24"""
	found = False
	for a, b in workingset:
		if a == n or b == n:
			found = True
			m = a if b == n else b
			if len(workingset) == 1:
				yield (a, b)
			else:
				for x in day24(m, workingset - {(a, b)}):
					yield (a, b) + x
	if not found:
		yield ()


def day24a(s):
	pairs = {tuple(map(int, a.split('/'))) for a in s.splitlines()}
	result = max(day24(0, pairs), key=sum)
	return sum(result)


def day24b(s):
	pairs = {tuple(map(int, a.split('/'))) for a in s.splitlines()}
	maxlen = max(map(len, day24(0, pairs)))
	result = max((a for a in day24(0, pairs) if len(a) == maxlen), key=sum)
	return sum(result)


def day25a(s):
	"""http://adventofcode.com/2017/day/25"""
	def decode(actions):
		return (
			0 if actions[0][-2] == '0' else 1,
			-1 if actions[1].endswith('left.') else 1,
			ord(actions[2][-2]))

	a, b = s.splitlines()[:2]
	start = ord(a[-2])
	checksum = int(b.split()[-2])
	states = {}
	for a in s.split('\n\n')[1:]:
		aa = a.splitlines()
		states[ord(aa[0][-2])] = {
				int(aa[1][-2]): decode(aa[2:5]),
				int(aa[5][-2]): decode(aa[6:9])}
	tape = {}
	pos = 0
	state = start
	for n in range(checksum):
		tape[pos], diff, state = states[state][tape.get(pos, 0)]
		pos += diff
	return sum(tape.values())


def day25b(s):
	"""There is no 25b puzzle."""


# -------8<-------- Tests  -------8<--------
def test1():
	assert day1a('1122') == 1 + 2
	assert day1a('1111') == 4
	assert day1a('1234') == 0
	assert day1a('91212129') == 9

	assert day1b('1212') == 6
	assert day1b('1221') == 0
	assert day1b('123425') == 4
	assert day1b('123123') == 12
	assert day1b('12131415') == 4


def test2():
	assert day2a(
		'5 1 9 5\n'
		'7 5 3\n'
		'2 4 6 8\n') == 8 + 4 + 6

	assert day2b(
		'5 9 2 8\n'
		'9 4 7 3\n'
		'3 8 6 5\n') == 4 + 3 + 2


def test3():
	assert day3a(1) == 0
	assert day3a(12) == 3
	assert day3a(23) == 2
	assert day3a(1024) == 31

	assert day3b(1) == 2
	assert day3b(2) == 4
	assert day3b(3) == 4
	assert day3b(4) == 5
	assert day3b(5) == 10


def test4():
	assert day4a('aa bb cc dd ee') == 1
	assert day4a('aa bb cc dd aa') == 0
	assert day4a('aa bb cc dd aaa') == 1

	assert day4b('abcde fghij') == 1
	assert day4b('abcde xyz ecdab') == 0
	assert day4b('a ab abc abd abf abj') == 1
	assert day4b('iiii oiii ooii oooi oooo') == 1
	assert day4b('oiii ioii iioi iiio') == 0


def test5():
	assert day5a('0\n3\n0\n1\n-3') == 5
	assert day5b('0\n3\n0\n1\n-3') == 10


def test6():
	assert day6a('0 2 7 0') == 5
	assert day6b('2 4 1 2') == 4


def test7():
	graph = (
			'pbga (66)\n'
			'xhth (57)\n'
			'ebii (61)\n'
			'havc (66)\n'
			'ktlj (57)\n'
			'fwft (72) -> ktlj, cntj, xhth\n'
			'qoyq (66)\n'
			'padx (45) -> pbga, havc, qoyq\n'
			'tknk (41) -> ugml, padx, fwft\n'
			'jptl (61)\n'
			'ugml (68) -> gyxo, ebii, jptl\n'
			'gyxo (61)\n'
			'cntj (57)\n')
	assert day7a(graph) == 'tknk'
	assert day7b(graph) == 60


def test8():
	x = (
			'b inc 5 if a > 1\n'
			'a inc 1 if b < 5\n'
			'c dec -10 if a >= 1\n'
			'c inc -20 if c == 10\n')
	assert day8a(x) == 1
	assert day8b(x) == 10


def test9():
	assert day9a('{}') == 1
	assert day9a('{{{}}}') == 1 + 2 + 3
	assert day9a('{{},{}}') == 1 + 2 + 2
	assert day9a('{{{},{},{{}}}}') == 1 + 2 + 3 + 3 + 3 + 4
	assert day9a('{<a>,<a>,<a>,<a>}') == 1
	assert day9a('{{<ab>},{<ab>},{<ab>},{<ab>}}') == 1 + 2 + 2 + 2 + 2
	assert day9a('{{<!!>},{<!!>},{<!!>},{<!!>}}') == 1 + 2 + 2 + 2 + 2
	assert day9a('{{<a!>},{<a!>},{<a!>},{<ab>}}') == 1 + 2
	# assert day9b('') == ...


def test10():
	assert day10a('3, 4, 1, 5', 5) == 3 * 4
	assert day10b('') == 'a2582a3a0e66e6e86e3812dcb672a272'
	assert day10b('AoC 2017') == '33efeb34ea91902bb2f59c9920caa6cd'
	assert day10b('1,2,3') == '3efbe78a8d82f29979031a4aa0b16a9d'
	assert day10b('1,2,4') == '63960835bcdc130f0b66d7ff4f6a5a8e'


def test11():
	assert day11a('ne,ne,ne') == 3
	assert day11a('ne,ne,sw,sw') == 0
	assert day11a('ne,ne,s,s') == 2
	assert day11a('se,sw,se,sw,sw') == 3

	assert day11b('ne,ne,ne') == 3
	assert day11b('ne,ne,sw,sw') == 2
	assert day11b('ne,ne,s,s') == 2
	assert day11b('se,sw,se,sw,sw') == 3


def test12():
	graph = (
			'0 <-> 2\n'
			'1 <-> 1\n'
			'2 <-> 0, 3, 4\n'
			'3 <-> 2, 4\n'
			'4 <-> 2, 3, 6\n'
			'5 <-> 6\n'
			'6 <-> 4, 5\n')
	assert day12a(graph) == 6
	assert day12b(graph) == 2


def test13():
	inp = ('0: 3\n'
			'1: 2\n'
			'4: 4\n'
			'6: 4\n')
	assert day13a(inp) == 0 * 3 + 6 * 4
	assert day13b(inp) == 10


def test14():
	assert day14a('flqrgnkx') == 8108
	assert day14b('flqrgnkx') == 1242


def test15():
	assert day15a('65\n8921\n', cycles=5) == 1
	assert day15b('65\n 8921\n', cycles=1056) == 1
	# assert day15a('65\n8921\n') == 588
	# assert day15b('65\n8921\n') == 309


def test16():
	assert day16a('s1,x3/4,pe/b', 5) == 'baedc'
	assert day16b('s1,x3/4,pe/b', 5, 2) == 'ceadb'


def test17():
	assert day17a('3') == 638


def test18():
	assert day18a(
			'set a 1\n'
			'add a 2\n'
			'mul a a\n'
			'mod a 5\n'
			'snd a\n'
			'set a 0\n'
			'rcv a\n'
			'jgz a -1\n'
			'set a 1\n'
			'jgz a -2\n') == 4
	assert day18b(
			'snd 1\n'
			'snd 2\n'
			'snd p\n'
			'rcv a\n'
			'rcv b\n'
			'rcv c\n'
			'rcv d\n') == 3


def test19():
	diagram = (
			'     |          \n'
			'     |  +--+    \n'
			'     A  |  C    \n'
			' F---|----E|--+ \n'
			'     |  |  |  D \n'
			'     +B-+  +--+ \n')
	assert day19a(diagram) == 'ABCDEF'
	assert day19b(diagram) == 38


def test20():
	assert day20a(
			'p=< 3,0,0>, v=< 2,0,0>, a=<-1,0,0>\n'
			'p=< 4,0,0>, v=< 0,0,0>, a=<-2,0,0>\n',
			iterations=10) == 0
	assert day20b(
			'p=<-6,0,0>, v=< 3,0,0>, a=< 0,0,0>\n'
			'p=<-4,0,0>, v=< 2,0,0>, a=< 0,0,0>\n'
			'p=<-2,0,0>, v=< 1,0,0>, a=< 0,0,0>\n'
			'p=< 3,0,0>, v=<-1,0,0>, a=< 0,0,0>\n',
			iterations=3) == 1


def test21():
	assert day21a(
			'../.# => ##./#../...\n'
			'.#./..#/### => #..#/..../..../#..#\n',
			iterations=2) == 12


def test22():
	grid = ('..#\n'
			'#..\n'
			'...\n')
	assert day22a(grid, iterations=70) == 41
	assert day22a(grid) == 5587
	assert day22b(grid, iterations=100) == 26
	# assert day22b(grid, iterations=10000000) == 2511944


def test23():
	assert day23b('') == 903


def test24():
	ex = (
			'0/2\n'
			'2/2\n'
			'2/3\n'
			'3/4\n'
			'3/5\n'
			'0/1\n'
			'10/1\n'
			'9/10\n')
	assert day24a(
			ex) == (0 + 1) + (1 + 10) + (10 + 9)
	assert day24b(ex) == 19


def test25():
	assert day25a("""\
Begin in state A.
Perform a diagnostic checksum after 6 steps.

In state A:
  If the current value is 0:
    - Write the value 1.
    - Move one slot to the right.
    - Continue with state B.
  If the current value is 1:
    - Write the value 0.
    - Move one slot to the left.
    - Continue with state B.

In state B:
  If the current value is 0:
    - Write the value 1.
    - Move one slot to the left.
    - Continue with state A.
  If the current value is 1:
    - Write the value 1.
    - Move one slot to the right.
    - Continue with state A.""") == 3


def benchmark():
	import timeit
	for n in range(1, 25 + 1):
		for part in 'ab':
			fun = 'day%d%s' % (n, part)
			time = timeit.timeit(
					'%s(inp)' % fun,
					setup='inp = open("i%d").read().rstrip("\\n")' % n,
					number=1,
					globals=globals())
			print('%s\t%5.2fs' % (fun, time))


if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
		benchmark()
	elif len(sys.argv) > 1 and sys.argv[1].startswith('day'):
		print(eval(sys.argv[1])(sys.stdin.read().rstrip('\n')))
	else:
		raise ValueError('unrecognized command')
