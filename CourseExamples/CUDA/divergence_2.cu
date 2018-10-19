/* This program shows branch divergence in warps, if you split the
 * threads in odd / even warps.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o divergence_2 divergence_2.cu
 *   clang --cuda-gpu-arch=sm_50 -o divergence_2 divergence_2.cu -lcudart
 *
 * Running:
 *   ./divergence_2
 *
 *
 * File: divergence_2.cu		Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>

#define BLOCKS_PER_GRID   1
#define THREADS_PER_BLOCK 64		/* two warps			*/


/* define macro to check the return value of a CUDA function		*/
#define CheckRetValueOfCudaFunction(val) \
  if (val != cudaSuccess) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, cudaGetErrorString (val)); \
    cudaDeviceReset (); \
    exit (EXIT_FAILURE); \
  }

__global__ void showDivergence (void);


int main (void)
{
  cudaError_t ret;			/* CUDA function return value   */

  showDivergence <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>> ();

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  return EXIT_SUCCESS;
}


__global__ void showDivergence (void)
{
  int warpNumber = threadIdx.x / warpSize;

  /* warpNumber can also be used in a switch-statment, if different
   * warps should do different work.
   */
  if ((warpNumber & 1) == 0)
  {
    /* even thread numbers						*/
    printf ("Thread %d in warp %d from thread block %d doing "
	    "work 1.\n",
	  threadIdx.x, warpNumber, blockIdx.x);
  }
  else
  {
    /* even thread numbers						*/
    printf ("Thread %d in warp %d from thread block %d doing "
	    "work 2.\n",
	  threadIdx.x, warpNumber, blockIdx.x);
  }
}
