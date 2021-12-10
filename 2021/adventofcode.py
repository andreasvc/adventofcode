"""Advent of Code 2021. http://adventofcode.com/2021 """
import os
import re
import sys
# import operator
# import itertools
# from functools import reduce
from collections import Counter  #, defaultdict
import numpy as np
# from numba import njit


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
			xd = 1 if x2 >= x1 else - 1
			yd = 1 if y2 >= y1 else - 1
			x, y = x1, y1
			for m in range(max(abs(x2 - x1), abs(y2 - y1)) + 1):
				grid[y, x] += 1
				x += xd * (x1 != x2)
				y += yd * (y1 != y2)
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
					print(n, char, scores[char], line)
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


def benchmark():
	import timeit
	for name in list(globals()):
		match = re.match(r'day(\d+)[ab]?', name)
		if match is not None and os.path.exists('i%s' % match.group(1)):
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
		with open('i' + sys.argv[1][3:].rstrip('ab') if len(sys.argv) == 2
				else sys.argv[2]) as inp:
			print(globals()[sys.argv[1]](inp.read().rstrip('\n')))
	else:
		print('unrecognized command.\n'
				'usage: python3 adventofcode.py day[1-25][ab] [input]\n'
				'or: python3 adventofcode.py benchmark')


if __name__ == '__main__':
	main()
