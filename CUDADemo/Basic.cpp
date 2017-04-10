#include <stdlib.h>
#include <iostream>
#include "RandomInts.h"
#include "printArray.h"

/*
__global__ -> indicates a function that runs on the device as is called from the host

*/
__global__ void mykernel(void) {

}

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

__global__ void addParallel(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void addThread(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

/*
<<<1,1>>> -> Triple angle brackets indicate a call from host to device
<<<N,1>>> -> N Blocks
<<<1,N>>> -> N threads

Memory Handling:
cudaMalloc()
cudaFree()
cudaMemcpy()
*/
int main(int argc, char *argv[]) {
	printf("Hello World!\n");

	//----------------------------------------------------

	mykernel << <1, 1 >> > ();

	//----------------------------------------------------

	int a, b, c;//Host
	int *d_a, *d_b, *d_c;//Device
	int intTypeSize = sizeof(int);

	cudaMalloc((void **)&d_a, intTypeSize);
	cudaMalloc((void **)&d_b, intTypeSize);
	cudaMalloc((void **)&d_c, intTypeSize);

	a = 3;
	b = 7;

	cudaMemcpy(d_a, &a, intTypeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, intTypeSize, cudaMemcpyHostToDevice);

	add << <1, 1 >> > (d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("C:[%i]", c);

	//----------------------------------------------------
	//Parallelization via Blocks
	int d, e, f;//Host
	int *d_d, *d_e, *d_f;//Device
	const int numParallel = 10;
	int size = numParallel * sizeof(int);

	cudaMalloc((void **)&d_d, size);
	cudaMalloc((void **)&d_e, size);
	cudaMalloc((void **)&d_f, size);

	d = (int *)malloc(size);
	e = (int *)malloc(size);
	f = (int *)malloc(size);

	random_ints(d, numParallel);
	random_ints(e, numParallel);

	cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e, e, size, cudaMemcpyHostToDevice);

	addParallel << <N, 1 >> > (d_d, d_e, d_f);

	cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost);

	printArray(f, numParallel);

	free(d);
	free(e);
	free(f);
	cudaFree(d_d);
	cudaFree(d_e);
	cudaFree(d_f);

	//----------------------------------------------------
	//Parallelization with threads
	//Almost identical to blocks
	//While not deminstrated in this example, threads are valuable because they are designed to sync and share between each other unlike blocks.
	int d, e, f;//Host
	int *d_d, *d_e, *d_f;//Device

	cudaMalloc((void **)&d_d, size);
	cudaMalloc((void **)&d_e, size);
	cudaMalloc((void **)&d_f, size);

	d = (int *)malloc(size);
	e = (int *)malloc(size);
	f = (int *)malloc(size);

	random_ints(d, numParallel);
	random_ints(e, numParallel);

	cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_e, e, size, cudaMemcpyHostToDevice);

	addThreads << <1, N >> > (d_d, d_e, d_f);

	cudaMemcpy(f, d_f, size, cudaMemcpyDeviceToHost);

	printArray(f, numParallel);

	free(d);
	free(e);
	free(f);
	cudaFree(d_d);
	cudaFree(d_e);
	cudaFree(d_f);

	//----------------------------------------------------

	return 0;
}