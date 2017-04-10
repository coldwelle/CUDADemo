/*

Input: A 1D Array
<[],[],[],[],[],[]>
with arbitrary numbers in each slot

Output: A 1D array where each index is the sum of each index within the input array that is within a radius of the output index.

So, if the radius is 2:
Output[X] = Input[X-2] + Input[X-1] + Input[X] + Input[X+1] + Input[X+2]

This will lead to unused positions at the edges where a full radius can't be achieved.

*/

#include <stdlib.h>
#include <iostream>
#include "RandomInts.h"
#include "printArray.h"

#define BLOCK_SIZE = 20;
#define RADIUS = 3;

__global__ void stencil_ld(int *in, int *out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + RADIUS;

	temp[lindex] in[gindex];
	if (threadIdx.x < RADIUS) {
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
	}

	__syncthreads();

	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
		result += temp[lindex + offset];
	}

	out[gindex] = result;
}

int main(int argc, char *argv[]) {
	int in, d_in;
	int out, d_out;
	int size = sizeof(int) * BLOCK_SIZE;

	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);

	in = (int *)malloc(size);
	out = (int *)malloc(size);

	random_ints(in, BLOCK_SIZE);

	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

	stencil << <1, BLOCK_SIZE >> > (in, out);

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	printArray(in);
	printf("\n");
	printArray(out);

	free(in);
	free(out);

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}