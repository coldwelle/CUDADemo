#include "RandomInts.h"
#include <random>

void random_ints(int *a, int N) {
	for (unsigned int cnt = 0; cnt < N; ++cnt) {
		a[cnt] = std::rand() % 100 + 1;
	}
}