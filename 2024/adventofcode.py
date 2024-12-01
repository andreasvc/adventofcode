"""Advent of Code 2024. http://adventofcode.com/2024 """
import sys
from collections import Counter
sys.path.append('..')
from common import main


def day1(s):
	data = [[int(a) for a in line.split()] for line in s.splitlines()]
	data = list(zip(*data))
	a, b = sorted(data[0]), sorted(data[1])
	cnt = Counter(b)
	result1 = sum(abs(x - y) for x, y in zip(a, b))
	result2 = sum(x * cnt[x] for x in a)
	return result1, result2


if __name__ == '__main__':
	main(globals())
