#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define RANGE 11.79

#define BLOCKS_PER_GRID 16
#define THREADS_PER_BLOCK 250

/*** TODO: insert the declaration of the kernel function below this line ***/
__global__ void vecGPU(float* ad, float* bd, float* cd, int calcs_per_thead, int width);
/**** end of the kernel declaration ***/


int main(int argc, char *argv[]){

	int n = 0; //number of elements in the arrays
	int i;  //loop index
	float *a, *b, *c; // The arrays that will be processed in the host.
	float *temp;  //array in host used in the sequential code.
	float *ad, *bd, *cd; //The arrays that will be processed in the device.
	clock_t start, end; // to meaure the time taken by a specific part of code
	
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
	}
		
	n = atoi(argv[1]);
	printf("Each vector will have %d elements\n", n);
	
	
	//Allocating the arrays in the host
	
	if( !(a = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}
	
	if( !(b = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array b\n");
	   exit(1);
	}
	
	if( !(c = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array c\n");
	   exit(1);
	}
	
	if( !(temp = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array temp\n");
	   exit(1);
	}
	
	//Fill out the arrays with random numbers between 0 and RANGE;
	srand((unsigned int)time(NULL));
	for (i = 0; i < n;  i++){
        a[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		b[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		c[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		temp[i] = c[i];
	}
	
    //The sequential part
	start = clock();
	for(i = 0; i < n; i++)
		temp[i] += a[i] * b[i];
	end = clock();
	printf("Total time taken by the sequential part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	printf("Total time taken by the sequential part = %d\n", (end - start)*1000);
    /******************  The start GPU part: Do not modify anything in main() above this line  ************/
	//The GPU part
	start = clock();
	
	/* TODO: in this part you need to do the following:
		1. allocate ad, bd, and cd in the device
		2. send a, b, and c to the device
		3. write the kernel, call it: vecGPU
		4. call the kernel (the kernel itself will be written at the comment at the end of this file), 
		   you need to decide about the number of threads, blocks, etc and their geometry.
		5. bring the cd array back from the device and store it in c array (declared earlier in main)
		6. free ad, bd, and cd
	*/
	int size = n * sizeof(float);

	cudaMalloc((void**) &ad, size);
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &bd, size);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &cd, size);
	cudaMemcpy(cd, c, size, cudaMemcpyHostToDevice);

	int calcs_per_thread = n / (BLOCKS_PER_GRID * THREADS_PER_BLOCK);

	// Kernal invocation
	vecGPU<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(ad, bd, cd, calcs_per_thread, n);

	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
	cudaFree(ad); 
	cudaFree(bd); 
	cudaFree(cd);

	end = clock();
	printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	/******************  The end of the GPU part: Do not modify anything in main() below this line  ************/
	
	//checking the correctness of the GPU part
	// for(i = 0; i < n; i++) {
	//   if(temp[i] != c[i])
	// 	printf("Element %d in the result array does not match the sequential version (%lf vs. %lf)\n", i, c[i], temp[i]);
	// }
	// Free the arrays in the host
	free(a); free(b); free(c); free(temp);

	return 0;
}

/**** TODO: Write the kernel itself below this line *****/
__global__ void vecGPU(float* ad, float* bd, float* cd, int calcs_per_thread, int width) {
	int global_id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int index = global_id * calcs_per_thread;
	int base_work = BLOCKS_PER_GRID * THREADS_PER_BLOCK * calcs_per_thread;
	int additional_work = width - base_work;

	for(int j = 0; j < calcs_per_thread; j++) {
		cd[index + j] += ad[index + j] * bd[index + j];
	}

	if(global_id < additional_work) {
		cd[base_work + global_id] += ad[base_work + global_id] * bd[base_work + global_id];
	}
}
