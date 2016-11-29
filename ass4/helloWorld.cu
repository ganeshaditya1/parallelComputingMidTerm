#include <stdio.h>

__global__ void kernel(void a)
{
	
}


int main()
{
	kernel<<<1, 1>>>();
	printf("Asd");
	return 0;
}