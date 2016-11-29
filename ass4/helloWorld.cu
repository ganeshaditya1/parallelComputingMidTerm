#include <stdio.h>

__global__ void kernel(int* a, int* b, int* c)
{
	*c = *a + *b;
}


int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	*a = 7;
	*b = 8;
	d_a = cudaMalloc((void **)&d_a, sizeof(int));
	d_b = cudaMalloc((void **)&d_b, sizeof(int));
	cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int), cudaMemcpyHostToDevice);
	kernel<<<1, 1>>>(d_a, d_b, d_c);
	cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Asd %d\n", *c);
	return 0;
}