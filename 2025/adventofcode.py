"""Advent of Code 2024. http://adventofcode.com/2024 """
import re
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


def day2(s):
	result1 = result2 = 0
	repetition = re.compile(r'([0-9]+)\1+$')
	for rng in s.split(','):
		a, b = map(int, rng.split('-'))
		for n in range(a, b + 1):
			ns = str(n)
			match = repetition.match(ns)
			if match:
				result2 += n
				if ns.count(match.group(1)) == 2:
					result1 += n
	return result1, result2


def day3(s):
	result1 = result2 = 0
	for bank in s.splitlines():
		result1 += max(int(a + b)
				for n, a in enumerate(bank)
					for b in bank[n + 1:])
		jolt = ''
		for n in range(12):
			digit = max(bank[:len(bank) - (12 - n - 1)])
			bank = bank[bank.index(digit) + 1:]
			jolt += digit
		result2 += int(jolt)
	return result1, result2


def day4(s):
	rolls = {(y, x)
			for y, line in enumerate(s.splitlines())
			for x, c in enumerate(line)
			if c == '@'}
	dirs = ((-1, 0), (1, 0), (0, -1), (0, 1),
			(-1, -1), (1, 1), (1, -1), (-1, 1))
	result1 = sum(
			sum((y + dy, x + dx) in rolls
				for dy, dx in dirs)
			< 4
			for y, x in rolls)
	result2 = 0
	while True:
		remove = {
			(y, x) for y, x in rolls
			if sum((y + dy, x + dx) in rolls
				for dy, dx in dirs) < 4}
		if not remove:
			break
		rolls -= remove
		result2 += len(remove)
	return result1, result2


def day5(s):
	...


def day6(s):
	...


def day7(s):
	...


def day8(s):
	...


def day9(s):
	...


def day10(s):
	...


def day11(s):
	...


def day12(s):
	...


if __name__ == '__main__':
	main(globals())
