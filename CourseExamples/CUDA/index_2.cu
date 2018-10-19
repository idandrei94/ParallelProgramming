/* A small program initializing arrays with different values.
 *
 * This version uses 1D grids and 1D blocks defined as "dim3"
 * (y- and z-directions must be "1").
 *
 * kernelConstant:	  using a constant
 * kernelBlockIdx:	  using blockIdx.x
 * kernelThreadIdx:	  using threadIdx.x
 * kernelGlobalThreadIdx: using the global thread index
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o index_2 index_2.cu
 *   clang --cuda-gpu-arch=sm_50 -o index_2 index_2.cu -lcudart
 *
 * Running:
 *   ./index_2
 *
 *
 * File: index_2.cu			Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCKS_PER_GRID_X   2
#define BLOCKS_PER_GRID_Y   1		/* must be "1"			*/
#define BLOCKS_PER_GRID_Z   1		/* must be "1"			*/
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 1		/* must be "1"			*/
#define THREADS_PER_BLOCK_Z 1		/* must be "1"			*/
#define VECTOR_SIZE	  BLOCKS_PER_GRID_X * THREADS_PER_BLOCK_X


__global__ void kernelConstant (int *a, const size_t vecSize);
__global__ void kernelBlockIdx (int *a, const size_t vecSize);
__global__ void kernelThreadIdx (int *a, const size_t vecSize);
__global__ void kernelGlobalThreadIdx (int *a, const size_t vecSize);



/* define macro to check the return value of a CUDA function		*/
#define CheckRetValueOfCudaFunction(val) \
  if (val != cudaSuccess) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, cudaGetErrorString (val)); \
    cudaDeviceReset (); \
    exit (EXIT_FAILURE); \
  }
    

int main (void)
{
  int aConstant[VECTOR_SIZE],		/* arrays on CPU		*/
      aBlockIdx[VECTOR_SIZE],
      aThreadIdx[VECTOR_SIZE],
      aGlobalThreadIdx[VECTOR_SIZE],
      *dev_aConstant,			/* array addresses on device	*/
      *dev_aBlockIdx,
      *dev_aThreadIdx,
      *dev_aGlobalThreadIdx;
  cudaError_t ret;			/* CUDA function return value	*/
  dim3 blocksPerGrid (BLOCKS_PER_GRID_X,
		      BLOCKS_PER_GRID_Y,
		      BLOCKS_PER_GRID_Z);
  dim3 threadsPerBlock (THREADS_PER_BLOCK_X,
  			THREADS_PER_BLOCK_Y,
			THREADS_PER_BLOCK_Z);

  /* check that the program uses only 1D grids and 1D blocks		*/
  assert ((BLOCKS_PER_GRID_Y == 1) && (BLOCKS_PER_GRID_Z == 1) &&
	  (THREADS_PER_BLOCK_Y == 1) && (THREADS_PER_BLOCK_Z == 1));
  
  /* allocate memory for all arrays on the GPU (device)                 */
  ret = cudaMalloc ((void **) &dev_aConstant,
                    VECTOR_SIZE * sizeof (int));
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMalloc ((void **) &dev_aBlockIdx,
                    VECTOR_SIZE * sizeof (int));
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMalloc ((void **) &dev_aThreadIdx,
                    VECTOR_SIZE * sizeof (int));
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMalloc ((void **) &dev_aGlobalThreadIdx,
                    VECTOR_SIZE * sizeof (int));
  CheckRetValueOfCudaFunction (ret);

  /* run all kernels concurrently                                       */
  kernelConstant <<<blocksPerGrid, threadsPerBlock>>>
    (dev_aConstant, VECTOR_SIZE);
  kernelBlockIdx <<<blocksPerGrid, threadsPerBlock>>>
    (dev_aBlockIdx, VECTOR_SIZE);
  kernelThreadIdx <<<blocksPerGrid, threadsPerBlock>>>
    (dev_aThreadIdx, VECTOR_SIZE);
  kernelGlobalThreadIdx <<<blocksPerGrid, threadsPerBlock>>>
    (dev_aGlobalThreadIdx, VECTOR_SIZE);

  /* copy initialized arrays back from the GPU to the CPU		*/
  ret = cudaMemcpy (aConstant, dev_aConstant,
		    VECTOR_SIZE * sizeof (int), cudaMemcpyDeviceToHost);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMemcpy (aBlockIdx, dev_aBlockIdx,
		    VECTOR_SIZE * sizeof (int), cudaMemcpyDeviceToHost);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMemcpy (aThreadIdx, dev_aThreadIdx,
		    VECTOR_SIZE * sizeof (int), cudaMemcpyDeviceToHost);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMemcpy (aGlobalThreadIdx, dev_aGlobalThreadIdx,
		    VECTOR_SIZE * sizeof (int), cudaMemcpyDeviceToHost);
  CheckRetValueOfCudaFunction (ret);

  /* print results							*/
  printf ("Initialization with a constant\n");
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    printf ("  %d", aConstant[i]);
  }
  printf ("\n\nInitialization with the blockIdx\n");
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    printf ("  %d", aBlockIdx[i]);
  }
  printf ("\n\nInitialization with the threadIdx\n");
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    printf ("  %d", aThreadIdx[i]);
  }
  printf ("\n\nInitialization with the global thread index\n");
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    printf ("  %d", aGlobalThreadIdx[i]);
  }
  printf ("\n\n");

  /* free allocated memory on the GPU					*/
  ret = cudaFree (dev_aConstant);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (dev_aBlockIdx);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (dev_aThreadIdx);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (dev_aGlobalThreadIdx);
  CheckRetValueOfCudaFunction (ret);

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  return EXIT_SUCCESS;
}


/* Initialize vector "a" with different values using GPU threads.
 *
 * Input:		vecSize		vector size
 * Output		a		initialized array
 * Return value:	none
 * Sideeffects:		none
 *
 */
__global__ void kernelConstant (int *a, const size_t vecSize)
{
  int idx = (int) (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx < (int) vecSize)
  {
    a[idx] = 9;
  }
}


__global__ void kernelBlockIdx (int *a, const size_t vecSize)
{
  int idx = (int) (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx < (int) vecSize)
  {
    a[idx] = (int) blockIdx.x;
  }
}


__global__ void kernelThreadIdx (int *a, const size_t vecSize)
{
  int idx = (int) (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx < (int) vecSize)
  {
    a[idx] = (int) threadIdx.x;
  }
}


__global__ void kernelGlobalThreadIdx (int *a, const size_t vecSize)

{
  int idx = (int) (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx < (int) vecSize)
  {
    a[idx] = idx;
  }
}
