"""Advent of Code 2022. http://adventofcode.com/2022 """
# import re
import sys
# import json
# import itertools
# from operator import lt, gt, eq
# from functools import reduce
from collections import defaultdict, Counter
# from heapq import heappush, heappop
import numpy as np
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
	def getstacks():
		stacks = {label: [] for label in labels}
		for line in crates[::-1]:
			for label, char in zip(labels, line[1::4]):
				if char != ' ':
					stacks[label].append(char)
		return stacks

	crates, ops = s.split('\n\n')
	*crates, labels = crates.splitlines()
	labels = labels[1::4]
	stacks1, stacks2 = getstacks(), getstacks()
	for line in ops.splitlines():
		_move, num, _from, a, _to, b = line.split()
		for n in range(int(num)):
			stacks1[b].append(stacks1[a].pop())
		tmp = stacks2[a][-int(num):]
		stacks2[a][-int(num):] = []
		stacks2[b].extend(tmp)
	part1 = ''.join(stacks1[label][-1] for label in labels)
	part2 = ''.join(stacks2[label][-1] for label in labels)
	return part1, part2


def day6(s):
	part1 = part2 = None
	for n in range(len(s)):
		if part1 is None and len(set(s[n:n + 4])) == 4:
			part1 = n + 4
		if len(set(s[n:n + 14])) == 14:
			part2 = n + 14
			return part1, part2


def day7(s):
	def traverse(path):
		total[path] = sizes[path] + sum(traverse(path + (subdir, ))
				for subdir in dirs[path])
		return total[path]

	n = 0
	lines = s.splitlines()
	sizes, dirs = defaultdict(int), defaultdict(list)
	cwd = ()
	while n < len(lines):
		cmd = lines[n].split()
		if cmd[1] == 'cd':
			if cmd[2] == '..':
				cwd = cwd[:-1]
			else:
				cwd = cwd + (cmd[2], )
		elif cmd[1] == 'ls':
			while n + 1 < len(lines) and lines[n + 1][0] != '$':
				n += 1
				cmd = lines[n].split()
				if cmd[0] == 'dir':
					dirs[cwd].append(cmd[1])
				else:
					sizes[cwd] += int(cmd[0])
		else:
			raise ValueError
		n += 1

	total = {}
	traverse(('/', ))
	part1 = sum(size for size in total.values() if size <= 100000)
	part2 = min(size for size in total.values()
			if 70000000 - total[('/', )] + size >= 30000000)
	return part1, part2


def day8(s):
	trees = np.array([[int(a) for a in line]
			for line in s.splitlines()], dtype=int)
	scenic = np.zeros(trees.shape, dtype=int)
	part1 = 0
	for n in range(trees.shape[0]):
		for m in range(trees.shape[1]):
			if (n == 0 or m == 0 or n + 1 == trees.shape[0]
					or m + 1 == trees.shape[1]):
				part1 += 1
			elif (trees[n, :m].max() < trees[n, m]
					or trees[n, m + 1:].max() < trees[n, m]
					or trees[:n, m].max() < trees[n, m]
					or trees[n + 1:, m].max() < trees[n, m]):
				part1 += 1
			up = down = left = right = 0
			for x in range(n - 1, -1, -1):
				left += 1
				if trees[x, m] >= trees[n, m]:
					break
			for x in range(n + 1, trees.shape[0]):
				right += 1
				if trees[x, m] >= trees[n, m]:
					break
			for y in range(m - 1, -1, -1):
				up += 1
				if trees[n, y] >= trees[n, m]:
					break
			for y in range(m + 1, trees.shape[1]):
				down += 1
				if trees[n, y] >= trees[n, m]:
					break
			scenic[n, m] = up * down * left * right
	part2 = scenic.max().max()
	return part1, part2


def _day9(s, knots):
	def dump():
		for y in range(15, -15, -1):
			for x in range(-15, 15):
				for r in range(knots):
					if rope[r][0] == x and rope[r][1] == y:
						print('H' if r == 0 else r, end='')
						break
				else:
					print('s' if x == 0 and y == 0 else '.', end='')
			print()
		print('tail at', rope[-1][0], rope[-1][1])
		print()

	visited = defaultdict(int)
	rope = [[0, 0] for _ in range(knots)]
	visited[0, 0] = 1
	# print('== Iinitial State ==\n')
	# dump()
	for line in s.splitlines():
		# print('==', line, '==\n')
		cmd, num = line.split()
		direction = 1 if cmd == 'U' or cmd == 'R' else -1
		coordinate = 1 if cmd == 'U' or cmd == 'D' else 0
		for _ in range(int(num)):
			rope[0][coordinate] += direction
			for r in range(1, knots):
				for c in range(2):
					if abs(rope[r][c] - rope[r - 1][c]) > 1:
						rope[r][c] += (1 if rope[r - 1][c] > rope[r][c]
								else -1)
						if rope[r][not c] != rope[r - 1][not c]:
							rope[r][not c] += (1 if rope[r - 1][not c]
									> rope[r][not c] else -1)
			visited[rope[-1][0], rope[-1][1]] = 1
			# dump()
	# for y in range(4, -1, -1):
	# 	print(''.join('#' if (x, y) in visited else '.' for x in range(6)))
	# print(visited)
	return len(visited)


def day9(s):
	return _day9(s, knots=2), _day9(s, knots=10)


def day10(s):
	part1 = idx = 0
	part2 = np.zeros((40, 7), dtype=int)
	x = 1
	for line in s.splitlines():
		if line == 'noop':
			num = 0
			delta = 1
		elif line.startswith('addx'):
			num = int(line.split()[1])
			delta = 2
		else:
			raise ValueError
		for _ in range(delta):
			part2[idx % 40, idx // 40] = 1 if abs((idx % 40) - x) <= 1 else 0
			idx += 1
			if (idx - 20) % 40 == 0:
				part1 += x * idx
		x += num
	for y in range(6):
		print(''.join('#' if part2[x, y] else '.' for x in range(40)))
	return part1


def _day11(s, relief, rounds):
	monkeys = {}
	counts = Counter()
	div = 1
	for chunk in s.split('\n\n'):
		name = int(chunk.splitlines()[0].split()[1].strip(':'))
		items, op, test, iftrue, iffalse = [
				line.split(': ')[1] for line in chunk.splitlines()[1:]]
		items = [int(a) for a in items.split(',')]
		op = eval(op.replace('new = ', 'lambda old: '))
		test = int(test.split()[-1])
		iftrue = int(iftrue.split()[-1])
		iffalse = int(iffalse.split()[-1])
		monkeys[name] = (items, op, test, iftrue, iffalse)
		div *= test
	for round in range(rounds):
		for name, (items, op, test, iftrue, iffalse) in monkeys.items():
			x = len(items)
			for n, item in enumerate(items):
				items[n] = op(item)
				if relief:
					items[n] //= 3
				else:
					items[n] %= div
				dest = iftrue if items[n] % test == 0 else iffalse
				monkeys[dest][0].append(items[n])
			items[:x] = []
			counts[name] += x
	(_, a), (_, b) = counts.most_common(2)
	return a * b


def day11(s):
	return _day11(s, True, 20), _day11(s, False, 10000)


if __name__ == '__main__':
	main(globals())
