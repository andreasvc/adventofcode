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


def day1a(s):
	return max(sum(int(a) for a in elf.splitlines()) for elf in s.split('\n\n'))


def day1b(s):
	return sum(sorted([sum(int(a) for a in elf.splitlines())
			for elf in s.split('\n\n')])[-3:])


if __name__ == '__main__':
	main(globals())
