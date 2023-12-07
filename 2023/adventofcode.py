"""Advent of Code 2023. http://adventofcode.com/2023 """
import re
import sys
from math import prod
# import json
import itertools
# from operator import lt, gt, eq
# from functools import cache, reduce
from collections import Counter  # defaultdict
# from functools import cmp_to_key
# from heapq import heappush, heappop
# from colorama import Fore, Style
import numpy as np
from numba import njit
from numba.typed import List
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


@njit
def _day5(n, maps):
	for m in maps:
		for a, b, c in m:
			if b <= n < b + c:
				n = a + n - b
				break
	return n


@njit
def day5b(seeds, maps):
	result2 = _day5(seeds[0], maps)
	for a, b in zip(seeds[::2], seeds[1::2]):
		for n in range(a, a + b):
			result2 = min(result2, _day5(n, maps))
	return result2


def day5(s):
	maps = s.split('\n\n')
	seeds = [int(a) for a in maps[0].split(':')[1].split()]
	maps = [[[int(a) for a in line.split()]
				for line in m.splitlines()[1:]]
			for m in maps[1:]]
	maps = [np.array(m, dtype=int) for m in maps]
	maps = List(maps)
	result2 = day5b(np.array(seeds, dtype=int), maps)
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
	def evaluate_joker(line, J=12):
		hand, _ = line
		newhands = [hand]
		if J in hand:
			jidxs = [n for n, a in enumerate(hand) if a == J]
			# for cmb in itertools.product(*[list(strengths.values()) for _ in jidxs]):
			for cmb in combs[len(jidxs)]:
				newhand = hand[:]
				for idx, new in zip(jidxs, cmb):
					newhand[idx] = new
				newhands.append(newhand)
		return min(evaluate(newhand, hand) for newhand in newhands)

	def evaluate(hand, orig):
		counts = sorted(Counter(hand).values(), reverse=True)
		return [types.index(counts)] + orig

	types = [[5],
			[4, 1],
			[3, 2],
			[3, 1, 1],
			[2, 2, 1],
			[2, 1, 1, 1],
			[1, 1, 1, 1, 1]]
	strengths = dict(zip('A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2'.split(', '), range(13)))
	hands = [[[strengths[a] for a in hand], int(bid)] for hand, bid in [line.split() for line in s.splitlines()]]
	result1 = sum(rank * bid for rank, (_hand, bid)
			in enumerate(sorted(hands, key=lambda x: evaluate(x[0], x[0]), reverse=True), 1))
	strengths = dict(zip('A, K, Q, T, 9, 8, 7, 6, 5, 4, 3, 2, J'.split(', '), range(13)))
	hands = [[[strengths[a] for a in hand], int(bid)] for hand, bid in [line.split() for line in s.splitlines()]]
	combs = {n: list(itertools.product(*[list(strengths.values()) for _ in range(n)]))
			for n in range(1, 6)}
	result2 = sum(rank * bid for rank, (_hand, bid)
			in enumerate(sorted(hands, key=evaluate_joker, reverse=True), 1))
	return result1, result2


if __name__ == '__main__':
	main(globals())
