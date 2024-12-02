"""Advent of Code 2024. http://adventofcode.com/2024 """
import sys
from collections import Counter
sys.path.append('..')
from common import main


def day1(s):
	data = [[int(a) for a in line.split()] for line in s.splitlines()]
	a, b = list(map(sorted, zip(*data)))
	cnt = Counter(b)
	result1 = sum(abs(x - y) for x, y in zip(a, b))
	result2 = sum(x * cnt[x] for x in a)
	return result1, result2


def day2(s):
	def safe(data):
		return all(a > b and 1 <= a - b <= 3 for a, b in zip(data, data[1:])
			) or all(a < b and 1 <= b - a <= 3 for a, b in zip(data, data[1:]))

	result1 = result2 = 0
	for line in s.splitlines():
		data = [int(a) for a in line.split()]
		result1 += safe(data)
		result2 += any(safe(data[:n] + data[n + 1:])
				for n in range(len(data)))
	return result1, result2


if __name__ == '__main__':
	main(globals())
