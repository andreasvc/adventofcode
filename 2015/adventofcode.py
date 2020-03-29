"""Advent of Code 2017. http://adventofcode.com/2017 """
import re
import sys
import array
from collections import Counter, defaultdict
import numpy as np


def day1a(s):
	"""http://adventofcode.com/2015/day/1"""
	return s.count('(') - s.count(')')


def day1b(s):
	cur = 0
	for n, a in enumerate(s, 1):
		if a == '(':
			cur += 1
		elif a == ')':
			cur -= 1
		if cur == -1:
			return n


def day2(s):
	inp = [tuple(int(a) for a in line.split('x'))
			for line in s.splitlines()]
	paper = ribbon = 0
	for l, w, h in inp:
		sides = [l * w, w * h, h * l]
		paper += 2 * sum(sides) + min(sides)
		ribbon += 2 * min((l + w, w + h, h + l)) + l * w * h
	return paper, ribbon


def day3(s):
	x = y = 0
	seen = {(0, 0)}
	for a in s:
		if a == '^':
			y -= 1
		elif a == 'v':
			y += 1
		elif a == '<':
			x -= 1
		elif a == '>':
			x += 1
		seen.add((x, y))
	return seen


def day3a(s):
	return len(day3(s))


def day3b(s):
	return len(day3(s[::2]) | day3(s[1::2]))


def day4a(s):
	return day4(s.encode('ascii'), 5 * '0')


def day4b(s):
	return day4(s.encode('ascii'), 6 * '0')


def day4(s, prefix):
	import hashlib
	n, x = 0, ''
	while not x.startswith(prefix):
		n += 1
		x = hashlib.md5(b'%s%d' % (s, n)).hexdigest()
	return n


def day5a(s):
	return sum(
			sum(a in 'aeiou' for a in line) >= 3
				and any(a == b for a, b in zip(line, line[1:]))
				and all(a not in line for a in ('ab', 'cd', 'pq', 'xy'))
			for line in s.splitlines())


def day5b(s):
	repeatpair = re.compile(r'(..).*\1')
	repeatletter = re.compile(r'(.).\1')
	return sum(
			repeatpair.search(line) is not None
				and repeatletter.search(line) is not None
			for line in s.splitlines())


def day6a(s):
	lights = np.zeros((1000, 1000), dtype=np.bool)
	for line in s.splitlines():
		fields = line.split()
		if line.startswith('turn'):
			x1, y1 = [int(a) for a in fields[2].split(',')]
			x2, y2 = [int(a) for a in fields[4].split(',')]
			if line.startswith('turn on'):
				lights[x1 - 1:x2, y1 - 1:y2] = 1
			elif line.startswith('turn off'):
				lights[x1 - 1:x2, y1 - 1:y2] = 0
		elif line.startswith('toggle'):
			x1, y1 = [int(a) for a in fields[1].split(',')]
			x2, y2 = [int(a) for a in fields[3].split(',')]
			lights[x1 - 1:x2, y1 - 1:y2] = lights[x1 - 1:x2, y1 - 1:y2] == 0
	return lights.sum().sum()


def day6b(s):
	lights = np.zeros((1000, 1000), dtype=np.int64)
	for line in s.splitlines():
		fields = line.split()
		if line.startswith('turn'):
			x1, y1 = [int(a) for a in fields[2].split(',')]
			x2, y2 = [int(a) for a in fields[4].split(',')]
			if line.startswith('turn on'):
				lights[x1 - 1:x2, y1 - 1:y2] += 1
			elif line.startswith('turn off'):
				lights[x1 - 1:x2, y1 - 1:y2] -= 1
				lights[lights < 0] = 0
		elif line.startswith('toggle'):
			x1, y1 = [int(a) for a in fields[1].split(',')]
			x2, y2 = [int(a) for a in fields[3].split(',')]
			lights[x1 - 1:x2, y1 - 1:y2] += 2
	return lights.sum().sum()


def day7(s):
	x = {'0': 0, '1': 1}
	lines = s.splitlines()
	while lines:
		for n, line in enumerate(lines):
			f = line.split()
			if f[1] == '->' and f[0].isnumeric():
				x[f[2]] = int(f[0])
			elif f[1] == '->' and f[0].isalpha() and f[0] in x:
				x[f[2]] = x[f[0]]
			elif f[1] == 'AND' and f[0] in x and f[2] in x:
				x[f[4]] = x[f[0]] & x[f[2]]
			elif f[1] == 'OR' and f[0] in x and f[2] in x:
				x[f[4]] = x[f[0]] | x[f[2]]
			elif f[1] == 'LSHIFT' and f[0] in x:
				x[f[4]] = x[f[0]] << int(f[2])
			elif f[1] == 'RSHIFT' and f[0] in x:
				x[f[4]] = x[f[0]] >> int(f[2])
			elif f[0] == 'NOT' and f[1] in x:
				x[f[3]] = 65536 + ~x[f[1]]
			else:
				continue
			lines.pop(n)
			break
	return x


def day7a(s):
	return day7(s)['a']


def day8a(s):
	return sum(
		len(l) - len(re.sub(r'\\\\|\\"|\\[x]..', '.', l[1:-1]))
		for l in s.splitlines())


def day8b(s):
	return sum(2 + l.count('"') + l.count('\\')
			for l in s.splitlines())


def day9(s, func):
	import itertools
	dists = {(f[0], f[2]): int(f[4]) for f in
			(l.split() for l in s.splitlines())}
	for a, b in list(dists):
		dists[b, a] = dists[a, b]
	places = {a for a, _ in dists}
	return func(
		(sum(dists[a, b] for a, b in zip(x, x[1:])), x)
		for x in itertools.permutations(places, len(places)))


def day9a(s):
	return day9(s, min)


def day9b(s):
	return day9(s, max)


# -------8<-------- Tests  -------8<--------

def test3():
	assert day3b('^v') == 3
	assert day3b('^>v<') == 3
	assert day3b('^v^v^v^v^v') == 11


def test7():
	inp = """123 -> x
456 -> y
x AND y -> d
x OR y -> e
x LSHIFT 2 -> f
y RSHIFT 2 -> g
NOT x -> h
NOT y -> i"""
	assert day7(inp) == {'0': 0, '1': 1,
			'd': 72,
			'e': 507,
			'f': 492,
			'g': 114,
			'h': 65412,
			'i': 65079,
			'x': 123,
			'y': 456,
			}


def test8():
	assert day8a(r'''""
"abc"
"aaa\"aaa"
"\x27"''') == 12


def benchmark():
	import timeit
	for n in range(1, 25 + 1):
		print('day%d' % n, end='')
		for part in 'ab':
			fun = 'day%d%s' % (n, part)
			time = timeit.timeit(
					'%s(inp)' % fun,
					setup='inp = open("i%d").read().rstrip("\\n")' % n,
					number=1,
					globals=globals())
			print('\t%5.2fs' % time, end='')
		print()


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
