"""Advent of Code 2023. http://adventofcode.com/2023 """
import re
import sys
from math import prod, lcm
# import json
import itertools
# from operator import lt, gt, eq
# from functools import cache, reduce
# from collections import Counter, defaultdict
# from functools import cmp_to_key
# from heapq import heappush, heappop
from colorama import Fore, Style
import numpy as np
# from numba import njit
# from numba.typed import List
sys.path.append('..')
from common import main


def day1(s):
	digits = ['one', 'two', 'three', 'four', 'five', 'six', 'seven',
			'eight', 'nine']
	x = re.compile(r'\d|' + '|'.join(digits))
	xx = re.compile(r'\d|' + '|'.join(digits)[::-1])
	conv = dict(zip(digits, [str(a) for a in range(1, 10)]))
	for a in range(10):
		conv[str(a)] = str(a)
	result1 = result2 = 0
	for line in s.splitlines():
		digits = re.findall(r'\d', line)
		result1 += int(digits[0] + digits[-1])
		digit1 = x.search(line).group()
		digit2 = xx.search(line[::-1]).group()[::-1]
		result2 += int(conv[digit1] + conv[digit2])
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
                        (y - 1, y, y + 1),
                        (x - 1, x, x + 1))}
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
							(y - 1, y, y + 1),
							(x - 1, x, x + 1))
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


def _day5(n, maps):
	for m in maps:
		for a, b, c in m:
			if b <= n < b + c:
				n = a + n - b
				break
	return n


def _day5b(seeds, maps):
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
	return result2


def day5(s):
	maps = s.split('\n\n')
	seeds = [int(a) for a in maps[0].split(':')[1].split()]
	maps = [[[int(a) for a in line.split()]
				for line in m.splitlines()[1:]]
			for m in maps[1:]]
	maps = [np.array(m, dtype=int) for m in maps]
	result2 = _day5b(np.array(seeds, dtype=int), maps)
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
			in [re.findall('\w\w\w', line) for line in lines[2:]]}
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
	...


if __name__ == '__main__':
	main(globals())
