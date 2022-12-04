"""Advent of Code 2022. http://adventofcode.com/2022 """
# import re
import sys
# import json
# import itertools
# from operator import lt, gt, eq
# from functools import reduce
# from collections import Counter, defaultdict
# from heapq import heappush, heappop
# import numpy as np
# from colorama import Fore, Style
sys.path.append('..')
from common import main


def day1(s):
	sums = sorted(sum(int(a) for a in elf.splitlines())
			for elf in s.split('\n\n'))
	return sums[-1], sum(sums[-3:])


def day2(s):
	rock, paper, scissors = lose, draw, _ = (1, 2, 3)
	part1 = part2 = 0
	for line in s.splitlines():
		a, b = line.split()
		a, b = ord(a) - 64, ord(b) - 87
		part1 += b + (3 if a == b else
				0 if (a == rock and b == scissors)
					or (a == paper and b == rock)
					or (a == scissors and b == paper)
				else 6)
		part2 += ((paper if a == scissors else scissors if a == rock else rock)
				if b == lose
				else a + 3 if b == draw
				else (scissors if a == paper else rock if a == scissors
					else paper) + 6)
	return part1, part2


def day3(s):
	part1 = part2 = 0
	lines = s.splitlines()
	for line in lines:
		half = len(line) // 2
		item = next(iter(set(line[:half]) & set(line[half:])))
		part1 += ord(item) - 96 if item.islower() else (ord(item) - 64 + 26)
	for l1, l2, l3 in zip(lines[::3], lines[1::3], lines[2::3]):
		item = next(iter(set(l1) & set(l2) & set(l3)))
		part2 += ord(item) - 96 if item.islower() else (ord(item) - 64 + 26)
	return part1, part2


def day4(s):
	part1 = part2 = 0
	for line in s.splitlines():
		a, b, c, d = map(int, line.replace(',', '-').split('-'))
		part1 += (a - c) * (b - d) <= 0
		part2 += a <= d and b >= c
	return part1, part2


def day5(s):
	...


if __name__ == '__main__':
	main(globals())
