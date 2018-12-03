#include <stdio.h>
#include <math.h>

int isprime(int n) {
	int sqrtn = sqrt(n);
	for (int m=2; m < sqrtn; m++) {
		if (n % m == 0)
			return 0;
	}
	return 1;
}

int main() {
	int a = 1, b = 0, c = 0, f = 0, h = 0;
	/* int d = 0, e = 0 */
	if (a == 0) {
		b = 84;
		c = 84;
	} else {
		b = 84 * 100 + 100000;
		c = b + 17000;
	}
	while (1) {
		/*
		f = 1;
		for (d = 2; d < b; d += 1) {
			for (e = 2; e < b; e += 1) {
				if (d * e == b)
					f = 0;
			}
		}
		*/
		f = isprime(b);
		if (f == 0)
			h += 1;
		if (b == c) {
			printf("%d\n", h);
			return 0;
		}
		b += 17;
	}
}
