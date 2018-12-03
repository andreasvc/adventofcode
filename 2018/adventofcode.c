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
	size_t size;
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
	size_t size;
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
	size_t size;
	int length;
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


typedef void (*funcdef)();

funcdef func[25] = {&day1, &day2, &day3};


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
