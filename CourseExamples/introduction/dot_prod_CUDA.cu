/* Compute the dot product of two vectors in parallel with CUDA.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o dot_prod_CUDA dot_prod_CUDA.cu
 *   clang --cuda-gpu-arch=sm_50 -o dot_prod_CUDA dot_prod_CUDA.cu \
 *	   -lcudart
 *
 * Running:
 *   ./dot_prod_CUDA
 *
 *
 * File: dot_prod_CUDA.cu			Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define EPS		  DBL_EPSILON	/* from float.h (2.2...e-16)	*/

#define VECTOR_SIZE	  10000000	/* vector size (10^8)		*/
#define THREADS_PER_BLOCK 256		/* must be a power of two	*/
#define BLOCKS_PER_GRID   32

/* Checks the return value of CUDA functions.				*/
#define CheckRetValueOfCudaFunction(val) \
  if (val != cudaSuccess) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, cudaGetErrorString (val)); \
    cudaDeviceReset (); \
    exit (EXIT_FAILURE); \
  }


/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double host_a[VECTOR_SIZE],	/* vectors for dot product	*/
	      host_b[VECTOR_SIZE];

__global__ void dot_prod (const double * __restrict__ a,
			  const double * __restrict__ b,
			  double * __restrict__ partial_sum);


int main (void)
{
  double sum,
	 partial_sum[BLOCKS_PER_GRID],	/* result of each thread block	*/
	 *dev_a, *dev_b,		/* vector addresses on device	*/
	 *dev_partial_sum;
  cudaError_t ret;			/* CUDA function return value	*/

  /* assert that THREADS_PER_BLOCK is a power of two	 		*/
  assert (fabs (THREADS_PER_BLOCK - exp2 (log2 (THREADS_PER_BLOCK)))
	  <= EPS);

  /* allocate memory for all vectors on the GPU (device)		*/
  ret = cudaMalloc ((void **) &dev_a, VECTOR_SIZE * sizeof (double));
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMalloc ((void **) &dev_b, VECTOR_SIZE * sizeof (double));
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMalloc ((void **) &dev_partial_sum,
		    BLOCKS_PER_GRID * sizeof (double));
  CheckRetValueOfCudaFunction (ret);
 
  /* initialize vectors						*/
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    host_a[i] = 2.0;
    host_b[i] = 3.0;
  }

  /* copy vectors to the GPU						*/
  ret = cudaMemcpy (dev_a, host_a, VECTOR_SIZE * sizeof (double),
		    cudaMemcpyHostToDevice);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMemcpy (dev_b, host_b, VECTOR_SIZE * sizeof (double),
		    cudaMemcpyHostToDevice);
  CheckRetValueOfCudaFunction (ret);

  /* compute partial dot products					*/
  dot_prod <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>
    (dev_a, dev_b, dev_partial_sum);

  /* copy result vector "dev_partial_sum" back from the GPU to the CPU	*/
  ret = cudaMemcpy (partial_sum, dev_partial_sum,
		    BLOCKS_PER_GRID * sizeof (double),
		    cudaMemcpyDeviceToHost);
  CheckRetValueOfCudaFunction (ret);

  /* compute "sum" of partial results					*/
  sum = 0.0;
  for (int i = 0; i < BLOCKS_PER_GRID; ++i)
  {
    sum += partial_sum[i];
  }

  /* free allocated memory on the GPU					*/
  ret = cudaFree (dev_a);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (dev_b);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (dev_partial_sum);
  CheckRetValueOfCudaFunction (ret);

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  printf ("sum = %e\n", sum);
  return EXIT_SUCCESS;
}


/* Compute dot product of vectors "a" and "b" and store partial
 * results of each block of threads into "partial_sum".
 *
 * Input:		a, b		vectors for dot product
 * Output		partial_sum	partial sums of each block
 * Return value:	none
 * Sideeffects:		none
 *
 */
__global__ void dot_prod (const double * __restrict__ a,
			  const double * __restrict__ b,
			  double * __restrict__ partial_sum)
{
  /* Use shared memory to store each thread's running sum. The
   * compiler will allocate a copy of shared variables for each
   * block of threads
   */
  __shared__ double cache[THREADS_PER_BLOCK];

  double temp = 0.0;
  int    cacheIdx = (int) threadIdx.x;
  
  for (int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
       tid < VECTOR_SIZE;
       tid += blockDim.x * gridDim.x)
  {
    temp += a[tid] * b[tid];
  }
  cache[cacheIdx] = temp;

  /* Ensure that all threads have completed, before you add up the
   * partial sums of each thread to the sum of the block
   */
  __syncthreads ();

  /* Each thread will add two values and store the result back to
   * "cache". We need "log_2 (THREADS_PER_BLOCK)" steps to reduce
   * all partial values to one block value. THREADS_PER_BLOCK must
   * be a power of two for this reduction.
   */
  for (int i = blockDim.x / 2; i > 0; i /= 2)
  {
    if (cacheIdx < i)
    {
      cache[cacheIdx] += cache[cacheIdx + i];
    }
    __syncthreads ();
  }
  /* store the partial sum of this thread block				*/
  if (cacheIdx == 0)
  {
    partial_sum[blockIdx.x] = cache[0];
  }
}
