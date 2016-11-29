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
	d_a = cudaMalloc(sizeof(int));
	d_b = cudaMalloc(sizeof(int));
	cudaMemcpy(d_a, a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int), cudaMemcpyHostToDevice);
	kernel<<<1, 1>>>(d_a, d_b, d_c);
	cudaMemcpy(d_c, c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Asd %d\n", *d_c);
	return 0;
}