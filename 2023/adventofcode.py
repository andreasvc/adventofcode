"""Advent of Code 2023. http://adventofcode.com/2023 """
import re
import sys
from math import prod
# import json
# import itertools
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


if __name__ == '__main__':
	main(globals())
