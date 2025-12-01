"""Advent of Code 2024. http://adventofcode.com/2024 """
import sys
sys.path.append('..')
from common import main


def day1(s):
	cur = 50
	result1 = result2 = 0
	for line in s.splitlines():
		dir, steps = line[0], int(line[1:])
		prev = cur
		if dir == 'L':
			cur = (cur - steps) % 100
			cond = (prev < cur and prev > 0)
		else:  # dir == 'R':
			cur = (cur + steps) % 100
			cond = prev > cur
		if cur == 0:
			result1 += 1
		if cond or cur == 0:
			result2 += 1
		result2 += steps // 100
	return result1, result2


if __name__ == '__main__':
	main(globals())
