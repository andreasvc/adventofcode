#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* http://c-faq.com/misc/bitsets.html */
#define BITSIZE                 (8 * sizeof(unsigned long))
#define BITMASK(b)              (1UL << ((b) % BITSIZE))
#define BITSLOT(b)              ((b) / BITSIZE)
#define SETBIT(a, b)            ((a)[BITSLOT(b)] |= BITMASK(b))
#define TOGGLEBIT(a, b)         ((a)[BITSLOT(b)] ^= BITMASK(b))
#define CLEARBIT(a, b)          ((a)[BITSLOT(b)] &= ~BITMASK(b))
#define TESTBIT(a, b)           ((a)[BITSLOT(b)] & BITMASK(b))
#define BITNSLOTS(nb)           (((nb) + BITSIZE - 1) / BITSIZE)

#define BITS 1000000
#define HALF (BITS / 2)

void day1() {
	int cur = 0;
	char *line = NULL;
	size_t size = 0;
	int inp[1000];
	int n;
	while (1) {
		if (getline(&line, &size, stdin) == -1) {
			break;
		}
		inp[n] = strtol(line, NULL, 10);
		cur += inp[n];
		n += 1;
	}
	printf("%d\n", cur);

	unsigned long seen[BITNSLOTS(BITS)];
	cur = 0;
	SETBIT(seen, cur + HALF);

	while (1) {
		for (int m=0; m<n; m++) {
			cur += inp[m];
			if (cur < -HALF || cur > HALF)
				abort();
			if (TESTBIT(seen, cur + HALF)) {
				printf("%d\n", cur);
				return;
			}
			SETBIT(seen, cur + HALF);
		}
	}
}


void day2() {
	char *line = NULL;
	size_t size = 0;
	int length;
	int twice = 0, thrice = 0;
	int counts[26];
	char lines[300][30];
	int m = 0;

	while (1) {
		length = getline(&line, &size, stdin);
		if (length == -1) {
			break;
		}
		strncpy(lines[m], line, 30);
		m += 1;
		for (int n=0; n < length; n++) {
			counts[line[n] - 97] += 1;
		}
		for (int n=0; n < 26; n++) {
			if (counts[n] == 2) {
				twice += 1;
				break;
			}
		}
		for (int n=0; n < 26; n++) {
			if (counts[n] == 3) {
				thrice += 1;
				break;
			}
		}
		for (int n=0; n < 26; n++) {
			counts[n] = 0;
		}
	}
	printf("%d\n", twice * thrice);

	for (int x=0; x < m; x++) {
		for (int y=x + 1; y < m; y++) {
			int diff = 0;
			for (int n=0; n<30; n++) {
				diff += lines[x][n] != lines[y][n];
			}
			if (diff == 1) {
				for (int n=0; n<30; n++) {
					if (lines[x][n] == lines[y][n]) {
						printf("%c", lines[x][n]);
					}
				}
				return;
			}
		}
	}
}


void day3() {
	char *line = NULL;
	char *ptr = NULL;
	size_t size = 0;
	char fabric[1000][1000] = {0};
	int ids[1500], xs[1500], ys[1500], widths[1500], heights[1500];
	int n = 0, count = 0;

	while (getline(&line, &size, stdin) != -1) {
		ids[n] = strtol(line + 1, &ptr, 10);
		xs[n] = strtol(ptr + 3, &ptr, 10);
		ys[n] = strtol(ptr + 1, &ptr, 10);
		widths[n] = strtol(ptr + 2, &ptr, 10);
		heights[n] = strtol(ptr + 1, &ptr, 10);
		n += 1;
	}
	for (int m=0; m<n; m++) {
		for (int a=xs[m]; a<xs[m] + widths[m]; a++) {
			for (int b=ys[m]; b<ys[m] + heights[m]; b++) {
				fabric[a][b] += 1;
			}
		}
	}
	for (int a=0; a<1000; a++) {
		for (int b=0; b<1000; b++) {
			count += fabric[a][b] > 1;
		}
	}
	printf("%d\n", count);

	for (int m=0; m<n; m++) {
		for (int a=xs[m]; a<xs[m] + widths[m]; a++) {
			for (int b=ys[m]; b<ys[m] + heights[m]; b++) {
				if (fabric[a][b] != 1) {
					goto breakloop;
				}
			}
		}
		printf("%d\n", ids[m]);
		return;
breakloop:
		continue;
	}
}


int _strcmp(const void *a, const void *b) {
	return strcmp(*(char * const *)a, *(char * const *)b);
}

void day4() {
	char *lines[2000] = {NULL};
	size_t size = 0;
	int numlines = 0;
	int minute, start, guard;
	int best = 0;
	int naps[10000][60] = {0};

	while (getline(&(lines[numlines]), &size, stdin) != -1) {
		size = 0;
		numlines += 1;
	}
	qsort(lines, numlines, sizeof(char *), _strcmp);
	for (int n=0; n<numlines; n++) {
		minute = strtol(lines[n] + 15, NULL, 10);
		if (lines[n][19] == 'G') { // Guard #...
			guard = strtol(lines[n] + 26, NULL, 10);
		} else if (lines[n][19] == 'f') { // falls asleep
			start = minute;
		} else if (lines[n][19] == 'w') {  // wakes up
			for (int a=start; a<minute; a++) {
				naps[guard][a] += 1;
			}
		}
	}
	for (int n=0; n<10000; n++) {
		int sum = 0;
		for (int a=0; a<60; a++) {
			sum += naps[n][a];
		}
		if (sum > best) {
			best = sum;
			guard = n;
		}
	}
	minute = 0;
	for (int a=0; a<60; a++) {
		minute = naps[guard][a] > naps[guard][minute] ? a : minute;
	}
	printf("%d\n", minute * guard);

	best = 0;
	for (int n=0; n<10000; n++) {
		int max = 0;
		for (int a=0; a<60; a++) {
			max = naps[n][a] > max ? naps[n][a] : max;
		}
		if (max > best) {
			best = max;
			guard = n;
		}
	}
	minute = 0;
	for (int a=0; a<60; a++) {
		minute = naps[guard][a] > naps[guard][minute] ? a : minute;
	}
	printf("%d\n", minute * guard);
}


void day5() {
	char *line = NULL;
	size_t size = 0;
	char result[10000] = {0};
	int m = 0;
	int length = getline(&line, &size, stdin);
	for (int n=0; n < length - 1; n++) {
		if (m && (line[n] ^ 32) == result[m - 1]) {
			m -= 1;
		} else {
			result[m] = line[n];
			m += 1;
		}
	}
	printf("%d\n", m);

	int best = m;
	int mm = 0;
	char result2[10000] = {0};
	for (int skip=97; skip < 97 + 26; skip++) {
		mm = 0;
		for (int n=0; n < m; n++) {
			if ((result[n] | 32) == skip) {
			} else if (mm && (result[n] ^ 32) == result2[mm - 1]) {
				mm -= 1;
			} else {
				result2[mm] = result[n];
				mm += 1;
			}
		}
		best = mm < best ? mm : best;
	}
	printf("%d\n", best);
}


typedef void (*funcdef)();

funcdef func[25] = {&day1, &day2, &day3, &day4, &day5};


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("usage: %s [1-25] < input\n", argv[0]);
		return 1;
	}
	int puzzle = strtol(argv[1], NULL, 10);
	if (puzzle < 1 || puzzle > 25) {
		printf("usage: %s [1-25] < input\n", argv[0]);
		return 1;
	}
	if (func[puzzle - 1] == NULL) {
		return 1;
	}
    func[puzzle - 1]();
	return 0;
}
