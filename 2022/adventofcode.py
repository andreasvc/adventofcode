"""Advent of Code 2021. http://adventofcode.com/2021 """
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
	strategies = [line.split() for line in s.splitlines()]
	strategies = [(ord(a) - 64, ord(b) - 87) for a, b in strategies]
	rock, paper, scissors = lose, draw, _ = (1, 2, 3)
	part1 = sum(b + (3 if a == b else
		0 if (a == rock and b == scissors)
			or (a == paper and b == rock)
			or (a == scissors and b == paper)
		else 6)
		for a, b in strategies)
	part2 = sum(
		(paper if a == scissors else scissors if a == rock else rock)
		if b == lose
		else a + 3 if b == draw
		else (scissors if a == paper else rock if a == scissors else paper) + 6
		for a, b in strategies)
	return part1, part2


if __name__ == '__main__':
	main(globals())
