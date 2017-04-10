#include <iostream>
#include "printArray.h"

void printArray(int *a, int N) {
	for (int cnt = 0; cnt < N; ++cnt) {
		printf("%i, ", a[cnt]);
	}
}