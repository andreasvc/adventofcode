"""Advent of Code 2017. http://adventofcode.com/2017 """
import re
import sys
import operator
import itertools
import functools
from collections import Counter, defaultdict
import numpy as np
from numba import njit


def day1a(s):
	"""http://adventofcode.com/2015/day/1"""
	return s.count('(') - s.count(')')


def day1b(s):
	cur = 0
	for n, a in enumerate(s, 1):
		if a == '(':
			cur += 1
		elif a == ')':
			cur -= 1
		if cur == -1:
			return n


def day2a(s):
	return day2(s)[0]


def day2b(s):
	return day2(s)[1]


def day2(s):
	inp = [tuple(int(a) for a in line.split('x'))
			for line in s.splitlines()]
	paper = ribbon = 0
	for length, width, height in inp:
		sides = [length * width, width * height, height * length]
		paper += 2 * sum(sides) + min(sides)
		ribbon += 2 * min((length + width, width + height, height + length)
				) + length * width * height
	return paper, ribbon


def day3(s):
	x = y = 0
	seen = {(0, 0)}
	for a in s:
		if a == '^':
			y -= 1
		elif a == 'v':
			y += 1
		elif a == '<':
			x -= 1
		elif a == '>':
			x += 1
		seen.add((x, y))
	return seen


def day3a(s):
	return len(day3(s))


def day3b(s):
	return len(day3(s[::2]) | day3(s[1::2]))


def day4a(s):
	return day4(s.encode('ascii'), 5 * '0')


def day4b(s):
	return day4(s.encode('ascii'), 6 * '0')


def day4(s, prefix):
	import hashlib
	n, x = 0, ''
	while not x.startswith(prefix):
		n += 1
		x = hashlib.md5(b'%s%d' % (s, n)).hexdigest()
	return n


def day5a(s):
	return sum(
			sum(a in 'aeiou' for a in line) >= 3
				and any(a == b for a, b in zip(line, line[1:]))
				and all(a not in line for a in ('ab', 'cd', 'pq', 'xy'))
			for line in s.splitlines())


def day5b(s):
	repeatpair = re.compile(r'(..).*\1')
	repeatletter = re.compile(r'(.).\1')
	return sum(
			repeatpair.search(line) is not None
				and repeatletter.search(line) is not None
			for line in s.splitlines())


def day6a(s):
	lights = np.zeros((1000, 1000), dtype=np.bool)
	for line in s.splitlines():
		fields = line.split()
		if line.startswith('turn'):
			x1, y1 = [int(a) for a in fields[2].split(',')]
			x2, y2 = [int(a) for a in fields[4].split(',')]
			if line.startswith('turn on'):
				lights[x1 - 1:x2, y1 - 1:y2] = 1
			elif line.startswith('turn off'):
				lights[x1 - 1:x2, y1 - 1:y2] = 0
		elif line.startswith('toggle'):
			x1, y1 = [int(a) for a in fields[1].split(',')]
			x2, y2 = [int(a) for a in fields[3].split(',')]
			lights[x1 - 1:x2, y1 - 1:y2] = lights[x1 - 1:x2, y1 - 1:y2] == 0
	return lights.sum().sum()


def day6b(s):
	lights = np.zeros((1000, 1000), dtype=np.int64)
	for line in s.splitlines():
		fields = line.split()
		if line.startswith('turn'):
			x1, y1 = [int(a) for a in fields[2].split(',')]
			x2, y2 = [int(a) for a in fields[4].split(',')]
			if line.startswith('turn on'):
				lights[x1 - 1:x2, y1 - 1:y2] += 1
			elif line.startswith('turn off'):
				lights[x1 - 1:x2, y1 - 1:y2] -= 1
				lights[lights < 0] = 0
		elif line.startswith('toggle'):
			x1, y1 = [int(a) for a in fields[1].split(',')]
			x2, y2 = [int(a) for a in fields[3].split(',')]
			lights[x1 - 1:x2, y1 - 1:y2] += 2
	return lights.sum().sum()


def day7(s, overrideb=None):
	x = {'0': 0, '1': 1}
	lines = s.splitlines()
	while lines:
		for n, line in enumerate(lines):
			f = line.split()
			if f[1] == '->' and f[0].isnumeric():
				if overrideb is not None and f[2] == 'b':
					x['b'] = overrideb
				else:
					x[f[2]] = int(f[0])
			elif f[1] == '->' and f[0].isalpha() and f[0] in x:
				x[f[2]] = x[f[0]]
			elif f[1] == 'AND' and f[0] in x and f[2] in x:
				x[f[4]] = x[f[0]] & x[f[2]]
			elif f[1] == 'OR' and f[0] in x and f[2] in x:
				x[f[4]] = x[f[0]] | x[f[2]]
			elif f[1] == 'LSHIFT' and f[0] in x:
				x[f[4]] = x[f[0]] << int(f[2])
			elif f[1] == 'RSHIFT' and f[0] in x:
				x[f[4]] = x[f[0]] >> int(f[2])
			elif f[0] == 'NOT' and f[1] in x:
				x[f[3]] = 65536 + ~x[f[1]]
			else:
				continue
			lines.pop(n)
			break
	return x


def day7a(s):
	return day7(s)['a']


def day7b(s):
	return day7(s, overrideb=day7(s)['a'])['a']


def day8a(s):
	return sum(
		len(line) - len(re.sub(r'\\\\|\\"|\\[x]..', '.', line[1:-1]))
		for line in s.splitlines())


def day8b(s):
	return sum(2 + line.count('"') + line.count('\\')
			for line in s.splitlines())


def day9(s, func):
	dists = {(f[0], f[2]): int(f[4]) for f in
			(line.split() for line in s.splitlines())}
	for a, b in list(dists):
		dists[b, a] = dists[a, b]
	places = {a for a, _ in dists}
	return func(
		sum(dists[a, b] for a, b in zip(x, x[1:]))
		for x in itertools.permutations(places, len(places)))


def day9a(s):
	return day9(s, min)


def day9b(s):
	return day9(s, max)


def day10a(s):
	return day10(s, 40)


def day10b(s):
	return day10(s, 50)


def day10(s, iterations):
	for n in range(iterations):
		s = ''.join(
				'%d%s' % (len(m.group(0)), m.group(1))
				for m in re.finditer(r'([0-9])\1*', s))
	return len(s)


def day11a(s):
	num = 0
	for n, a in enumerate(s[::-1]):  # convert to number
		num += (ord(a) - 97) * 26 ** n
	while True:
		num += 1
		n, s = num, ''
		while n:  # convert number back to string
			n, m = divmod(n, 26)
			s = chr(97 + m) + s
		if (any(ord(c) - ord(b) == ord(b) - ord(a) == 1 for a, b, c in
				zip(s, s[1:], s[2:]))
				and 'i' not in s and 'o' not in s and 'l' not in s
				and len(set(re.findall(r'(.)\1', s))) >= 2):
			return s


def day11b(s):
	return day11a(day11a(s))


def day12a(s):
	return sum(int(a) for a in re.findall('[-0-9]+', s))


def day12b(s):
	def traverse(data):
		if isinstance(data, dict):
			if 'red' in data.values():
				return 0
			return sum(traverse(a) for a in data.values())
		elif isinstance(data, list):
			return sum(traverse(a) for a in data)
		return data if isinstance(data, int) else 0

	import json
	return traverse(json.loads(s))


def day13a(s):
	return day13(s)


def day13b(s):
	return day13(s, includeself=True)


def day13(s, includeself=False):
	data = {(fields[0], fields[-1]):
				int(fields[3]) if fields[2] == 'gain' else -int(fields[3])
			for fields in
			(line.rstrip('.').split() for line in s.splitlines())}
	names = {a for a, _ in data}
	if includeself:
		names.add('yourself')
	return max(
			sum(data.get((a, b), 0) for a, b in zip(perm, perm[1:]))
			for perm in
				(perm + perm[0:1] + perm[::-1] for perm
				in itertools.permutations(names, len(names))))


def day14a(s):
	data = {fields[0]: (int(fields[3]), int(fields[6]), int(fields[-2]))
			for fields in (line.split() for line in s.splitlines())}
	duration = 2503
	return max(
			((duration // (flytime + resttime)) * speed * flytime)
			+ min(duration % (flytime + resttime), flytime) * speed
			for name, (speed, flytime, resttime) in data.items())


def day14b(s, duration=2503):
	data = {fields[0]: (int(fields[3]), int(fields[6]), int(fields[-2]))
			for fields in (line.split() for line in s.splitlines())}
	points = dict.fromkeys(data, 0)
	distance = dict.fromkeys(data, 0)
	state = {name: ('flying', flytime)
			for name, (_, flytime, _) in data.items()}
	for n in range(duration):
		for name, (activity, timeleft) in state.items():
			speed, flytime, resttime = data[name]
			if activity == 'flying':
				distance[name] += speed
			if timeleft > 1:
				state[name] = activity, timeleft - 1
			else:
				state[name] = (('resting', resttime) if activity == 'flying'
						else ('flying', flytime))
		for name in distance:
			if distance[name] == max(distance.values()):
				points[name] += 1
	return max(points.values())


def day15a(s):
	def squashandmult(seq):
		return 0 if (seq < 0).any() else seq.prod()

	data = {name:
			np.array([int(a.split()[1]) for a in fields.split(', ')
				if not a.startswith('calories')], dtype=np.int)
			for name, fields in
			(line.split(':', 1) for line in s.splitlines())}
	return max(
			squashandmult(sum(data[elem] for elem in comb))
			for comb in itertools.combinations_with_replacement(
				data.keys(), 100))


def day15b(s):
	def squashandmult(seq):
		return 0 if (seq < 0).any() else seq.prod()

	data = {name:
			np.array([int(a.split()[1]) for a in fields.split(', ')],
				dtype=np.int)
			for name, fields in
			(line.split(':', 1) for line in s.splitlines())}
	return max(
			squashandmult(sum(data[elem][:-1] for elem in comb))
			for comb in itertools.combinations_with_replacement(
				data.keys(), 100)
			if sum(data[elem][-1] for elem in comb) == 500)


def day16a(s):
	data = [line.split(': ', 1)[1].split(', ')
			for line in s.splitlines()]
	aunt = """children: 3
cats: 7
samoyeds: 2
pomeranians: 3
akitas: 0
vizslas: 0
goldfish: 5
trees: 3
cars: 2
perfumes: 1"""
	return [n for n, props in enumerate(data, 1)
			if all(a in aunt for a in props)].pop()


def day16b(s):
	data = [{a.split(':')[0]: int(a.split(':')[1])
				for a in line.split(': ', 1)[1].split(', ')}
			for line in s.splitlines()]
	aunt = {a.split(':')[0]: int(a.split(':')[1])
			for a in """children: 3
cats: 7
samoyeds: 2
pomeranians: 3
akitas: 0
vizslas: 0
goldfish: 5
trees: 3
cars: 2
perfumes: 1""".splitlines()}
	return [n for n, props in enumerate(data, 1)
			if all(
				aunt[a] < b if a in ('cats', 'trees')
				else (aunt[a] > b if a in ('pomeranians', 'goldfish')
				else aunt[a] == b)
				for a, b in props.items())].pop()


def day17a(s, goal=150):
	data = [int(a) for a in s.splitlines()]
	return sum(sum(comb) == goal
			for n in range(1, len(data) + 1)
			for comb in itertools.combinations(data, n))


def day17b(s, goal=150):
	data = [int(a) for a in s.splitlines()]
	lens = [len(comb)
			for n in range(1, len(data) + 1)
				for comb in itertools.combinations(data, n)
			if sum(comb) == goal]
	return lens.count(min(lens))


def day18a(s):
	data = np.array([[a == '#' for a in line]
			for line in s.splitlines()], dtype=np.bool)
	new = np.zeros(data.shape)
	for n in range(100):
		for x in range(data.shape[0]):
			for y in range(data.shape[1]):
				neighbors = data[
							max(x - 1, 0):min(x + 2, data.shape[0]),
							max(y - 1, 0):min(y + 2, data.shape[1])].sum()
				if data[x, y]:
					new[x, y] = 3 <= neighbors <= 4
				else:
					new[x, y] = neighbors == 3
		data, new = new, data
	return data.sum().sum()


def day18b(s):
	data = np.array([[a == '#' for a in line]
			for line in s.splitlines()], dtype=np.bool)
	new = np.zeros(data.shape)
	for n in range(100):
		data[0, 0] = data[data.shape[0] - 1, 0] = 1
		data[0, data.shape[1] - 1] = 1
		data[data.shape[0] - 1, data.shape[1] - 1] = 1
		for x in range(data.shape[0]):
			for y in range(data.shape[1]):
				neighbors = data[
							max(x - 1, 0):min(x + 2, data.shape[0]),
							max(y - 1, 0):min(y + 2, data.shape[1])].sum()
				if data[x, y]:
					new[x, y] = 3 <= neighbors <= 4
				else:
					new[x, y] = neighbors == 3
		data, new = new, data
	data[0, 0] = data[data.shape[0] - 1, 0] = 1
	data[0, data.shape[1] - 1] = 1
	data[data.shape[0] - 1, data.shape[1] - 1] = 1
	return data.sum().sum()


def day19a(s):
	data = [line.split(' => ') for line in s.splitlines() if '=>' in line]
	molecule = s.splitlines()[-1]
	results = set()
	for a, b in data:
		for m in re.finditer(a, molecule):
			results.add(molecule[:m.start()] + b + molecule[m.end():])
	return len(results)


def day19b(s):
	data = [line.split(' => ') for line in s.splitlines() if '=>' in line]
	cur = {s.splitlines()[-1]}
	generation = 0
	while True:
		generation += 1
		new = set()
		for mol in cur:
			for a, b in data:
				for m in re.finditer(b, mol):
					new.add(mol[:m.start()] + a + mol[m.end():])
		cur = {min(new, key=lambda x: len(x))}
		if 'e' in cur:
			return generation


def day20a(s):
	goal = int(s)
	houses = [0] * 1000000
	for n in range(1, len(houses)):
		for m in range(n, len(houses), n):
			houses[m] += n * 10
	for n in range(len(houses)):
		if houses[n] >= goal:
			return n


def day20b(s):
	goal = int(s)
	houses = [0] * 1000000
	for n in range(1, len(houses) + 1):
		for m in range(n, min(len(houses), 50 * n + 1), n):
			houses[m] += n * 11
	for n in range(len(houses)):
		if houses[n] >= goal:
			return n


def day21(s):
	def play(me, boss):
		while True:
			boss['Hit Points'] -= max(1, me['Damage'] - boss['Armor'])
			if boss['Hit Points'] <= 0:
				return True
			me['Hit Points'] -= max(1, boss['Damage'] - me['Armor'])
			if me['Hit Points'] <= 0:
				return False

	def parse(items):
		return [np.array([int(a) for a in line.split()[1:]], dtype=int)
				for line in items.splitlines()]

	boss = dict(zip(
			[line.split(':')[0] for line in s.splitlines()],
			[int(line.split(':')[1]) for line in s.splitlines()]))
	me = {'Hit Points': 100, 'Damage': 0, 'Armor': 0}
	# Weapons:    Cost  Damage  Armor
	weapons = parse("""Dagger        8     4       0
	Shortsword   10     5       0
	Warhammer    25     6       0
	Longsword    40     7       0
	Greataxe     74     8       0""")

	# Armor:      Cost  Damage  Armor
	armor = parse("""Leather      13     0       1
	Chainmail    31     0       2
	Splintmail   53     0       3
	Bandedmail   75     0       4
	Platemail   102     0       5""")

	# Rings:      Cost  Damage  Armor
	rings = parse("""Damage+1    25     1       0
	Damage+2    50     2       0
	Damage+3   100     3       0
	Defense+1   20     0       1
	Defense+2   40     0       2
	Defense+3   80     0       3""")
	candidates = []
	for weapon in weapons:
		for arm in armor:
			for comb in itertools.chain(
					[()],
					itertools.combinations(rings, 1),
					itertools.combinations(rings, 2)):
				candidate = weapon + arm
				if len(comb):
					candidate += comb[0]
				if len(comb) == 2:
					candidate += comb[1]
				candidates.append(candidate)
	candidates.sort(key=lambda x: x[0])
	for cand in candidates:
		x = me.copy()
		x['Damage'] += cand[1]
		x['Armor'] += cand[2]
		if play(x, boss.copy()):
			ans1 = cand[0]
			break
	for cand in candidates[::-1]:
		x = me.copy()
		x['Damage'] += cand[1]
		x['Armor'] += cand[2]
		if not play(x, boss.copy()):
			ans2 = cand[0]
			break
	return ans1, ans2


def play22(spells, boss, hard=False):
	myhp = 50
	mana = 500
	manaspent = 0
	armor = np.zeros(2 * len(spells), dtype=int)
	damage = np.zeros(2 * len(spells), dtype=int)
	recharge = np.zeros(2 * len(spells), dtype=int)
	cost = {'missile': 53, 'drain': 73, 'shield': 113, 'poison': 173,
			'recharge': 229}
	n = 0
	for spell in spells:
		# our turn
		if hard:
			myhp -= 1
			if myhp <= 0:
				return False
		if mana < cost[spell]:
			return False
		mana -= cost[spell]
		manaspent += cost[spell]
		if spell == 'missile':
			damage[n] += 4
		elif spell == 'drain':
			damage[n] += 2
			myhp += 2
		elif spell == 'shield' and armor[n + 1] == 0:
			armor[n + 1:n + 7] += 7
		elif spell == 'poison' and damage[n + 1] == 0:
			damage[n + 1:n + 7] += 3
		elif spell == 'recharge' and recharge[n + 1] == 0:
			recharge[n + 1:n + 6] += 101
		else:
			return False  # spell was applied while still active
		boss['Hit Points'] -= damage[n]
		if boss['Hit Points'] <= 0:
			return manaspent
		mana += recharge[n]
		n += 1
		# boss turn
		mana += recharge[n]
		boss['Hit Points'] -= damage[n]
		if boss['Hit Points'] <= 0:
			return manaspent
		myhp -= max(1, boss['Damage'] - armor[n])
		if myhp <= 0:
			return False
		n += 1


def day22(s):
	boss = dict(zip(
			[line.split(':')[0] for line in s.splitlines()],
			[int(line.split(':')[1]) for line in s.splitlines()]))
	spells = 'missile drain shield poison recharge'.split()
	ans1, ans2 = [], []
	for size in range(9):
		for comb in itertools.product(spells, repeat=size):
			m = play22(comb, boss.copy())
			if m:
				ans1.append(m)
			m = play22(comb, boss.copy(), hard=True)
			if m:
				ans2.append(m)
				break
	return min(ans1), min(ans2)


def _day23(s, a=0):
	pc = 0
	state = {'a': a, 'b': 0}
	prog = s.splitlines()
	while 0 <= pc < len(prog):
		op, arg = prog[pc].split(None, 1)
		if ',' in arg:
			arg, offset = arg.split(',')
		if op == 'hlf':
			state[arg] //= 2
		elif op == 'tpl':
			state[arg] *= 3
		elif op == 'inc':
			state[arg] += 1
		elif op == 'jmp':
			pc += int(arg)
			continue
		elif op == 'jie':
			if (state[arg] & 1) == 0:
				pc += int(offset)
				continue
		elif op == 'jio':
			if state[arg] == 1:
				pc += int(offset)
				continue
		pc += 1
	return state['b']


def day23(s):
	return _day23(s, 0), _day23(s, 1)


def _day24(s, groups=3):
	weights = [int(a) for a in s.splitlines()]
	goal = sum(weights) // groups
	for size in range(2, len(weights)):
		result = [comb for comb in itertools.combinations(weights, size)
				if sum(comb) == goal]
		if result:
			break
	return min([functools.reduce(operator.mul, a, 1) for a in result])


def day24(s):
	return _day24(s, 3), _day24(s, 4)


def day25(s):
	goalcol = int(s.split()[-1].strip('.'))
	goalrow = int(s.split()[-3].strip(','))
	col = row = 1
	cur = 20151125
	while True:
		cur = cur * 252533 % 33554393
		if row > 1:
			row -= 1
			col += 1
		else:
			row = col + 1
			col = 1
		if col == goalcol and row == goalrow:
			return cur


# -------8<-------- Tests  -------8<--------

def test3():
	assert day3b('^v') == 3
	assert day3b('^>v<') == 3
	assert day3b('^v^v^v^v^v') == 11


def test7():
	inp = """123 -> x
456 -> y
x AND y -> d
x OR y -> e
x LSHIFT 2 -> f
y RSHIFT 2 -> g
NOT x -> h
NOT y -> i"""
	assert day7(inp) == {'0': 0, '1': 1,
			'd': 72,
			'e': 507,
			'f': 492,
			'g': 114,
			'h': 65412,
			'i': 65079,
			'x': 123,
			'y': 456,
			}


def test8():
	assert day8a(r'''""
"abc"
"aaa\"aaa"
"\x27"''') == 12


def test14b():
	s = """\
Comet can fly 14 km/s for 10 seconds, but then must rest for 127 seconds.
Dancer can fly 16 km/s for 11 seconds, but then must rest for 162 seconds."""
	assert day14b(s, 1000) == 689


def benchmark():
	import timeit
	for n in range(1, 25 + 1):
		print('day%d' % n, end='')
		for part in 'ab':
			fun = 'day%d%s' % (n, part)
			time = timeit.timeit(
					'%s(inp)' % fun,
					setup='inp = open("i%d").read().rstrip("\\n")' % n,
					number=1,
					globals=globals())
			print('\t%5.2fs' % time, end='')
		print()


def main():
	if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
		benchmark()
	elif len(sys.argv) > 1 and sys.argv[1].startswith('day'):
		with open('i' + sys.argv[1][3:].rstrip('ab') if len(sys.argv) == 2
				else sys.argv[2]) as inp:
			print(globals()[sys.argv[1]](inp.read().rstrip('\n')))
	else:
		raise ValueError('unrecognized command. '
				'usage: python3 adventofcode.py day[1-25][ab] [input]'
				'or: python3 adventofcode.py benchmark')


if __name__ == '__main__':
	main()
