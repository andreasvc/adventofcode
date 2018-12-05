"""Advent of Code 2018. http://adventofcode.com/2018 """
import re
import sys
import datetime
from itertools import cycle
from collections import Counter, defaultdict
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
