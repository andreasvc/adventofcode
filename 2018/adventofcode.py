"""Advent of Code 2018. http://adventofcode.com/2018 """
import re
import sys
import datetime
from itertools import cycle
from collections import Counter, defaultdict, deque
import numpy as np


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


if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
		benchmark()
	elif len(sys.argv) > 1 and sys.argv[1].startswith('day'):
		print(globals()[sys.argv[1]](sys.stdin.read().rstrip('\n')))
	else:
		raise ValueError('unrecognized command. '
				'usage: python3 adventofcode.py day[1-25][ab] < input'
				'or: python3 adventofcode.py benchmark')
