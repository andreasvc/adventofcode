"""Advent of Code 2018. http://adventofcode.com/2018 """
import re
import sys
import datetime
from itertools import cycle
from collections import Counter, defaultdict, deque
import numpy as np
from numba import njit


def day1a(s):
	return sum(int(line) for line in s.splitlines())


def day1b(s):
	cur, seen = 0, set()
	for line in cycle(s.splitlines()):
		seen.add(cur)
		cur += int(line)
		if cur in seen:
			return cur


def day2a(s):
	twice = thrice = 0
	for line in s.splitlines():
		a = Counter(line)
		twice += 2 in a.values()
		thrice += 3 in a.values()
	return twice * thrice


def day2b(s):
	lines = s.splitlines()
	for n, line in enumerate(lines):
		for line2 in lines[n + 1:]:
			if sum(x != y for x, y in zip(line, line2)) == 1:
				return ''.join(x for x, y in zip(line, line2) if x == y)


def day3(s):
	fabric = np.zeros((1000, 1000), dtype=np.int8)
	for line in s.splitlines():
		id, x, y, width, height = map(int, re.findall(r'\d+', line))
		fabric[x:x + width, y:y + height] += 1
	return fabric


def day3a(s):
	fabric = day3(s)
	return (fabric > 1).sum().sum()


def day3b(s):
	fabric = day3(s)
	for line in s.splitlines():
		id, x, y, width, height = map(int, re.findall(r'\d+', line))
		if (fabric[x:x + width, y:y + height] == 1).all().all():
			return id


def day4(s):
	events = []
	for line in s.splitlines():
		date, event = line.lstrip('[').split('] ')
		date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M')
		events.append((date, event))
	naps = defaultdict(lambda: np.zeros(60, dtype=int))
	for date, event in sorted(events):
		if 'Guard' in event:
			guard = int(event.split()[1][1:])
		elif 'asleep' in event:
			start = date
		elif 'wakes' in event:
			naps[guard][start.minute:date.minute] += 1
	return naps


def day4a(s):
	naps = day4(s)
	guard = max(naps, key=lambda x: naps[x].sum())
	return guard * naps[guard].argmax()


def day4b(s):
	naps = day4(s)
	guard = max(naps, key=lambda x: naps[x].max())
	return guard * naps[guard].argmax()


def day5(s):
	result = bytearray()
	for a in s:
		if result and a ^ 32 == result[-1]:
			result.pop()
		else:
			result.append(a)
	return result


def day5a(s):
	return len(day5(bytearray(s, 'ascii')))


def day5b(s):
	s = day5(bytearray(s, 'ascii'))
	return min(len(day5(bytearray(x for x in s if x | 32 != a)))
			for a in range(ord('a'), ord('a') + 26))


def day6a(s):
	from scipy.spatial import cKDTree
	X = np.array([[int(a) for a in line.split(',')]
			for line in s.splitlines() if line.strip()])
	tree = cKDTree(X)
	width, height = X[:, 0].max() + 1, X[:, 1].max() + 1
	borderqueries = [(x, y) for x in range(width) for y in (0, height - 1)]
	borderqueries += [(x, y) for x in (0, width - 1) for y in range(height)]
	infinite = set(tree.query(borderqueries, p=1, k=1)[1])
	xv, yv = np.meshgrid(np.arange(1, width - 1), np.arange(1, height - 1))
	queries = np.dstack([xv, yv]).reshape(-1, 2)
	dists, inds = tree.query(queries, p=1, k=2)
	return next(b for a, b in Counter(inds[dists[:, 0] != dists[:, 1], 0]
			).most_common() if a not in infinite)


def day6b(s):
	from scipy.spatial import distance_matrix
	X = np.array([[int(a) for a in line.split(',')]
			for line in s.splitlines() if line.strip()])
	width, height = X[:, 0].max() + 1, X[:, 1].max() + 1
	xv, yv = np.meshgrid(np.arange(width), np.arange(height))
	queries = np.dstack([xv, yv]).reshape(-1, 2)
	dists = distance_matrix(X, queries, p=1)
	return (dists.sum(axis=0) < 10000).sum()


def day7a(s):
	def visit(node):
		if node in seen:
			return
		for m in sorted({b for a, b in constraints if a == node},
				reverse=True):
			visit(m)
		seen.add(node)
		result.insert(0, node)

	constraints = {(a.split()[1], a.split()[7]) for a in s.splitlines()}
	steps = sorted({a for pair in constraints for a in pair})
	result = []
	seen = set()
	while steps:
		node = steps.pop()
		if node in seen:
			continue
		visit(node)
	return ''.join(result)


def day7b(s):
	queue = list(day7a(s))
	constraints = {(a.split()[1], a.split()[7]) for a in s.splitlines()}
	time = 0
	numworkers = 5
	duration = 60
	workers = [None for _ in range(numworkers)]
	done = []
	while True:
		for n, worker in enumerate(workers):
			if worker is not None and worker[0] == time:
				done.append(worker[1])
				workers[n] = None
				if not queue:
					return time
		for n, worker in enumerate(workers):
			if worker is None and queue:
				for m, task in enumerate(queue):
					if all(a in done for a, b in constraints if b == task):
						queue.pop(m)
						workers[n] = (time + duration + ord(task) - 64, task)
						break
		time += 1


def day8a(s):
	def getnode(i):
		numchildren = inp[i]
		nummd = inp[i + 1]
		i += 2
		md = []
		for _ in range(numchildren):
			i, b = getnode(i)
			md.extend(b)
		md.extend(inp[i:i + nummd])
		return i + nummd, md

	inp = [int(a) for a in s.split()]
	_, md = getnode(0)
	return sum(md)


def day8b(s):
	def getnode(i):
		numchildren = inp[i]
		nummd = inp[i + 1]
		i += 2
		values = []
		for _ in range(numchildren):
			i, b = getnode(i)
			values.append(b)
		if numchildren == 0:
			result = sum(inp[i:i + nummd])
		else:
			result = sum(values[a - 1] for a in inp[i:i + nummd]
					if a - 1 < len(values))
		return i + nummd, result

	inp = [int(a) for a in s.split()]
	_, result = getnode(0)
	return result


def day9a(s):
	players = int(s.split()[0])
	last = int(s.split()[-2])
	circle = deque([0])
	scores = [0] * (players + 1)
	for marble in range(1, last, 23):
		for n in range(marble, marble + 22):
			circle.rotate(2)
			circle.append(n)
		circle.rotate(-7)
		marble += 22
		scores[marble % players] += marble + circle.pop()
	return max(scores)


def day9b(s):
	players = int(s.split()[0])
	last = int(s.split()[-2])
	return day9a('%d players; last marble is worth %d points' % (
			players, last * 100))


def day10(s):
	inp = np.array([
		[int(a) for a in re.findall(r'-?\d+', line)]
		for line in s.splitlines()], dtype=int)
	pos, vel = inp[:, :2], inp[:, 2:]
	minwidth = minheight = 99999999
	for n in range(9999999):
		pos += vel
		width = pos[:, 0].max() - pos[:, 0].min()
		height = pos[:, 1].max() - pos[:, 1].min()
		if width > minwidth or height > minheight:
			pos -= vel
			break
		minwidth, minheight = width, height
	out = [['.'] * (minwidth + 1) for _ in range(minheight + 1)]
	x1, y1 = pos[:, 0].min(), pos[:, 1].min()
	for x, y in pos:
		out[y - y1][x - x1] = '#'
	out = '\n'.join(''.join(line) for line in out)
	return out, n


def day10a(s):
	return day10(s)[0]


def day10b(s):
	return day10(s)[1]


def day11(s):
	def powerlevel(serial, x, y):
		rackid = x + 10
		result = rackid * y
		result += serial
		result *= rackid
		result = (result // 100) % 10
		return result - 5

	serial = int(s)
	buf = np.zeros((301, 301), dtype=int)
	for x in range(1, 301):
		for y in range(1, 301):
			buf[x, y] = powerlevel(serial, x, y)
	return serial, buf


def day11a(s):
	serial, buf = day11(s)
	return '%d,%d' % max(((x, y)
			for x in range(1, 301 - 3)
				for y in range(1, 301 - 3)),
			key=lambda a: buf[a[0]:a[0] + 3, a[1]:a[1] + 3].sum())


@njit
def _day11b(serial, buf):
	n, m = 0, (0, 0, 0)
	for size in range(1, 301):
		print(size)
		for x in range(1, 301 - size):
			for y in range(1, 301 - size):
				nn = buf[x:x + size, y:y + size].sum()
				if nn > n:
					n, m = nn, (x, y, size)
					print(n, m)
	return m


def day11b(s):
	return '%d,%d,%d' % _day11b(*day11(s))


def day12a(s):
	padding = 25
	state = np.array([0] * padding + [a == '#'
			for a in s.splitlines()[0].split(': ')[1]] + [0] * padding,
			dtype=np.bool)
	rules = np.array([[a == '#' for a in line.replace(' => ', '')]
			for line in s.splitlines()[2:]],
			dtype=np.bool)
	print(' 0: %s' % ''.join('.#'[a] for a in state))
	for gen in range(1, 21):
		newstate = np.zeros(len(state), dtype=np.bool)
		for n in range(2, len(state) - 2):
			for rule in rules:
				if (state[n - 2:n + 3] == rule[:5]).all():
					newstate[n] = rule[5]
					break
		state = newstate
		print('%2d: %s' % (gen, ''.join('.#'[a] for a in state)))
	for n, a in enumerate(state, -padding):
		if a:
			print(n)
	return sum(n for n, a in enumerate(state, -padding) if a)


def day13(s):
	state = [[a for a in line] for line in s.splitlines()]
	tracks = [line.copy() for line in state]
	carts = []
	firstcollision = lastcart = None
	for y in range(len(state)):
		for x in range(len(state[0])):
			if tracks[y][x] in '<>^v':
				carts.append((y, x, 0))
				tracks[y][x] = '-' if tracks[y][x] in '<>' else '|'
	turns = {
			'^': {'|': '^', '/': '>', '\\': '<', '+': '<^>'},
			'>': {'-': '>', '/': '^', '\\': 'v', '+': '^>v'},
			'v': {'|': 'v', '/': '<', '\\': '>', '+': '>v<'},
			'<': {'-': '<', '/': 'v', '\\': '^', '+': 'v<^'},
			}
	while True:
		for n, (y, x, direction) in enumerate(carts):
			if direction == -1:
				continue
			cur = state[y][x]
			state[y][x] = tracks[y][x]
			if cur == '^':
				y -= 1
			elif cur == '>':
				x += 1
			elif cur == 'v':
				y += 1
			elif cur == '<':
				x -= 1
			if state[y][x] not in '-|+/\\':
				direction = -1
			elif tracks[y][x] == '+':
				state[y][x] = turns[cur][tracks[y][x]][direction]
				direction = (direction + 1) % 3
			else:
				state[y][x] = turns[cur][tracks[y][x]]
			carts[n] = (y, x, direction)
			if direction == -1:
				if firstcollision is None:
					firstcollision = '%d,%d' % (x, y)
				state[y][x] = tracks[y][x]
				for m, (ay, ax, _) in enumerate(carts):
					if ay == y and ax == x:
						carts[m] = (ay, ax, -1)
		if sum(direction != -1 for _, _, direction in carts) == 1:
			for y, x, direction in carts:
				if direction != -1:
					lastcart = '%d,%d' % (x, y)
					return firstcollision, lastcart
		carts.sort()


def day13a(s):
	return day13(s)[0]


def day13b(s):
	return day13(s)[1]


def day14a(s):
	num = int(s)
	states = a, b = [3, 7]
	na, nb = 0, 1
	while True:
		x = states[na] + states[nb]
		if x >= 10:
			states.append(x // 10)
			states.append(x % 10)
		else:
			states.append(x)
		na = (na + a + 1) % len(states)
		nb = (nb + b + 1) % len(states)
		a, b = states[na], states[nb]
		if len(states) >= num + 10:
			return ''.join(str(x) for x in states[num:num + 10])


def day14b(s):
	pattern = [int(a) for a in s]
	states = a, b = [3, 7]
	na, nb = 0, 1
	while True:
		x = states[na] + states[nb]
		if x >= 10:
			states.append(x // 10)
			states.append(x % 10)
		else:
			states.append(x)
		na = (na + a + 1) % len(states)
		nb = (nb + b + 1) % len(states)
		a, b = states[na], states[nb]
		if states[-7:-1] == pattern:
			return len(states) - 7
		if states[-6:] == pattern:
			return len(states) - 6


def day16(s):
	samples, program = s.split('\n\n\n')
	before = []
	after = []
	instruction = []
	for chunk in samples.split('\n\n'):
		a, b, c = chunk.splitlines()
		a = [int(x) for x in a.split(':')[1].strip(' []').split(', ')]
		c = [int(x) for x in c.split(':')[1].strip(' []').split(', ')]
		before.append(a)
		after.append(c)
		b = [int(x) for x in b.split()]
		instruction.append(b)
	possible = []
	for before, (op, val1, val2, out), after in zip(before, instruction, after):
		opcodes = set()
		# immediate only
		if val1 == after[out]:
			opcodes.add('seti')
		# immediate, register
		if val2 < 4 and out < 4:
			if (val1 > before[val2]) == after[out]:
				opcodes.add('gtir')
			if (val1 == before[val2]) == after[out]:
				opcodes.add('eqir')
		# register, (immediate)
		if val1 < 4 and out < 4:
			if (before[val1] + val2) == after[out]:
				opcodes.add('addi')
			if (before[val1] * val2) == after[out]:
				opcodes.add('muli')
			if (before[val1] & val2) == after[out]:
				opcodes.add('bani')
			if (before[val1] | val2) == after[out]:
				opcodes.add('bori')
			if (before[val1] > val2) == after[out]:
				opcodes.add('gtri')
			if (before[val1] == val2) == after[out]:
				opcodes.add('eqri')
			if before[val1] == after[out]:
				opcodes.add('setr')
		# register, register
		if val1 < 4 and val2 < 4 and out < 4:
			if before[val1] + before[val2] == after[out]:
				opcodes.add('addr')
			if before[val1] * before[val2] == after[out]:
				opcodes.add('mulr')
			if before[val1] & before[val2] == after[out]:
				opcodes.add('banr')
			if before[val1] | before[val2] == after[out]:
				opcodes.add('borr')
			if (before[val1] > before[val2]) == after[out]:
				opcodes.add('gtrr')
			if (before[val1] == before[val2]) == after[out]:
				opcodes.add('eqrr')
		possible.append(opcodes)
	return possible, instruction


def day16a(s):
	possible, _ = day16(s)
	return sum(1 for a in possible if len(a) >= 3)


def day16b(s):
	possible, instruction = day16(s)
	opcodetbl = {}
	ops = {op for op, _, _, _ in instruction}
	while not all(len(a) == 1 for a in possible):
		for op in ops:
			x = [a for a, (op1, _, _, _)
					in zip(possible, instruction) if op1 == op]
			result = x[0]
			result.intersection_update(*x[1:])
			if len(result) == 1:
				opcodetbl[next(iter(result))] = op
				for a, (op1, _, _, _) in zip(possible, instruction):
					if op1 == op:
						a &= result
					else:
						a -= result
	opcodes = sorted(opcodetbl, key=opcodetbl.get)
	reg = [0] * 4
	_, program = s.split('\n\n\n')
	for line in program.splitlines():
		if not line.strip():
			continue
		op, val1, val2, out = [int(a) for a in line.split()]
		op = opcodes[op]
		if op == 'seti':
			reg[out] = val1
		elif op == 'gtir':
			reg[out] = int(val1 > reg[val2])
		elif op == 'eqir':
			reg[out] = int(val1 == reg[val2])
		elif op == 'gtri':
			reg[out] = int(reg[val1] > val2)
		elif op == 'eqri':
			reg[out] = int(reg[val1] == val2)
		elif op == 'gtrr':
			reg[out] = int(reg[val1] > reg[val2])
		elif op == 'eqrr':
			reg[out] = int(reg[val1] == reg[val2])
		elif op == 'addi':
			reg[out] = reg[val1] + val2
		elif op == 'muli':
			reg[out] = reg[val1] * val2
		elif op == 'bani':
			reg[out] = reg[val1] & val2
		elif op == 'bori':
			reg[out] = reg[val1] | val2
		elif op == 'setr':
			reg[out] = reg[val1]
		elif op == 'addr':
			reg[out] = reg[val1] + reg[val2]
		elif op == 'mulr':
			reg[out] = reg[val1] * reg[val2]
		elif op == 'banr':
			reg[out] = reg[val1] & reg[val2]
		elif op == 'borr':
			reg[out] = reg[val1] | reg[val2]
	return reg[0]


def benchmark():
	import timeit
	for name in list(globals()):
		match = re.match(r'day(\d+)[ab]', name)
		if match is not None:
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
		inp = sys.stdin if len(sys.argv) == 2 else open(sys.argv[2])
		print(globals()[sys.argv[1]](inp.read().rstrip('\n')))
	else:
		raise ValueError('unrecognized command. '
				'usage: python3 adventofcode.py day[1-25][ab] < input'
				'or: python3 adventofcode.py benchmark')


if __name__ == '__main__':
	main()
