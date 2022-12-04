import os
import re
import sys
import colorama


def colortime(time, fmt='%7.3f'):
	"""Returns colored timing."""
	text = fmt % time
	if time > 1.0:
		return colorama.Fore.RED + text + colorama.Fore.RESET
	elif time > 0.1:
		return colorama.Fore.YELLOW + text + colorama.Fore.RESET
	elif time < 0.01:
		return colorama.Fore.GREEN + text + colorama.Fore.RESET
	return text


def benchmark(glb):
	import timeit
	total = 0.0
	for name in list(glb):
		match = re.match(r'day(\d+)[ab]?', name)
		if match is not None and os.path.exists('i%s' % match.group(1)):
			timer = timeit.Timer(
					'%s(inp)' % name,
					setup='inp = open("i%s").read().rstrip("\\n")'
						% match.group(1),
					# number=1,
					globals=glb)
			number, time = timer.autorange()
			time /= number
			total += time
			print('%s\t%ss' % (name, colortime(time)))
	print('total:  %ss' % colortime(total))


def main(glb):
	"""CLI. Pass `globals()` as argument."""
	if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
		benchmark(glb)
	elif len(sys.argv) > 1 and sys.argv[1].startswith('day'):
		with open('i' + sys.argv[1][3:].rstrip('ab') if len(sys.argv) == 2
				else sys.argv[2]) as inp:
			print(glb[sys.argv[1]](inp.read().rstrip('\n')))
	else:
		print('unrecognized command.\n'
				'usage: python3 %s day[1-25][ab] [input]\n'
				'or: python3 %s benchmark' % (sys.argv[0], sys.argv[0]))
