"""Advent of Code 2023. http://adventofcode.com/2023 """
import re
import sys
from math import prod
# import json
import itertools
# from operator import lt, gt, eq
# from functools import reduce
# from collections import defaultdict, Counter
# from functools import cmp_to_key
# from heapq import heappush, heappop
# import numpy as np
# from colorama import Fore, Style
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
	xmax, ymax = len(grid[0]), len(grid)
	result1 = 0
	for y, line in enumerate(grid):
		for match in re.finditer(r'\d+', line):
			context = ''
			a, b = match.span()
			if y > 0:
				context += grid[y - 1][max(0, a - 1):min(ymax, b + 1)]
			context += line[max(0, a - 1):min(xmax, b + 1)]
			if y < ymax - 1:
				context += grid[y + 1][max(0, a - 1):min(ymax, b + 1)]
			if re.search(r'[^.\d]', context) is not None:
				result1 += int(match.group())
	result2 = 0
	for y, line in enumerate(grid):
		for match in re.finditer(r'\*', line):
			x = match.start()
			nums = set()
			for yy, xx in itertools.product(
					(y - 1, y, y + 1), (x - 1, x, x + 1)):
				if 0 <= yy < ymax and 0 <= xx < xmax:
					if grid[yy][xx].isdigit():
						nums.add([int(match.group())
								for match in re.finditer(r'\d+', grid[yy])
								if match.start() <= xx < match.end()][0])
			if len(nums) == 2:
				result2 += prod(nums)
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

if __name__ == '__main__':
	main(globals())
