#include <stdlib.h>
#include <iostream>
#include "RandomInts.h"
#include "printArray.h"

__global__ void add(int *a, int *b, int *c) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

int main(int argc, char *argv[]) {
	int a, b, c;//Host
	int *d_a, *d_b, *d_c;//Device
	int size = N * sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	random_ints(a, N);
	random_ints(b, N);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add << <N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	printArray(c);

	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}