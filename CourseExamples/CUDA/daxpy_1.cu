/* Simplified implementation of the DAXPY subprogram (double
 * precision alpha x plus y) from the Basic Linear Algebra
 * Subprogram library (BLAS). This version supports an arbitrary
 * number of threads per thread block and thread blocks per grid,
 * so that the kernel must iterate through the index space of the
 * vector to compute the results for all vector elements.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o daxpy_1 daxpy_1.cu
 *   clang --cuda-gpu-arch=sm_50 -o daxpy_1 daxpy_1.cu -lcudart
 *
 * Running:
 *   ./daxpy_1
 *
 *
 * File: daxpy_1.cu			Author: S. Gross
 * Date: 14.02.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define	VECTOR_SIZE 100000000		/* vector size (10^8)		*/
#define ALPHA	    5.0			/* scalar alpha			*/
#define EPS	    DBL_EPSILON		/* from float.h (2.2...e-16)	*/
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID   32


/* define macro to check the return value of a CUDA function		*/
#define CheckRetValueOfCudaFunction(val) \
  if (val != cudaSuccess) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, cudaGetErrorString (val)); \
    cudaDeviceReset (); \
    exit (EXIT_FAILURE); \
  }


/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double host_x[VECTOR_SIZE],
	      host_y[VECTOR_SIZE];

__global__ void daxpy (const int n, const double alpha,
		       const double * __restrict__ x,
		       double * __restrict__ y);


int main (void)
{
  double      *dev_x, *dev_y,		/* vector on device		*/
	      tmp_y0;			/* temporary value		*/
  float       cudaTime;			/* CUDA elapsed time		*/
  int	      tmp_diff;			/* temporary value		*/
  time_t      start_wall, end_wall;	/* start/end time (wall clock)	*/
  clock_t     cpu_time;			/* used cpu time		*/
  cudaEvent_t startEvent, stopEvent;	/* start/end event (GPU)	*/
  cudaError_t ret;			/* CUDA function return value	*/


  /* allocate memory for all vectors on the GPU (device)		*/
  ret = cudaMalloc ((void **) &dev_x, VECTOR_SIZE * sizeof (double));
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMalloc ((void **) &dev_y, VECTOR_SIZE * sizeof (double));
  CheckRetValueOfCudaFunction (ret);

  /* Initialize both vectors. The daxpy function computes
   * y = alpha * x + y. With the following initialization we get
   * constant values for the resulting vector.
   * new_y[i] = alpha * x[i] + y[i]
   *	      = alpha * i + alpha * (VECTOR_SIZE - i)
   *	      = alpha * VECTOR_SIZE
   */
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    host_x[i] = (double) i;
    host_y[i] = ALPHA * (double) (VECTOR_SIZE - i);
  }

  /* copy vectors to the GPU						*/
  ret = cudaMemcpy (dev_x, host_x, VECTOR_SIZE * sizeof (double),
		    cudaMemcpyHostToDevice);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMemcpy (dev_y, host_y, VECTOR_SIZE * sizeof (double),
		    cudaMemcpyHostToDevice);
  CheckRetValueOfCudaFunction (ret);

  /* compute "y = alpha * x + y" and measure computation time		*/
  ret = cudaEventCreate (&startEvent);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaEventCreate (&stopEvent);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaEventRecord (startEvent, 0); /* default stream 0		*/
  CheckRetValueOfCudaFunction (ret);
  start_wall = time (NULL);
  cpu_time   = clock ();
  daxpy <<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>
    (VECTOR_SIZE, ALPHA, dev_x, dev_y);
  ret = cudaEventRecord (stopEvent, 0);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaEventSynchronize (stopEvent);
  CheckRetValueOfCudaFunction (ret);
  cpu_time = clock () - cpu_time;
  end_wall = time (NULL);
  ret = cudaEventElapsedTime (&cudaTime, startEvent, stopEvent);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaEventDestroy (startEvent);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaEventDestroy (stopEvent);
  CheckRetValueOfCudaFunction (ret);

  /* copy result vector "y" back from the GPU to the CPU		*/
  ret = cudaMemcpy (host_y, dev_y, VECTOR_SIZE * sizeof (double),
		    cudaMemcpyDeviceToHost);
  CheckRetValueOfCudaFunction (ret);

  /* Check result. All elements should have the same value.		*/
  tmp_y0   = host_y[0];
  tmp_diff = 0;
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    if (fabs (tmp_y0 - host_y[i]) > EPS)
    {
      tmp_diff++;
    }
  }
  if (tmp_diff == 0)
  {
    printf ("Computation was successful. y[0] = %6.2f\n", host_y[0]);
  }
  else
  {
    printf ("Computation was not successful. %d values differ.\n",
	    tmp_diff);
  }

  /* show computation time						*/
  printf ("elapsed time      cpu time      GPU elapsed time\n"
	  "    %6.2f s      %6.2f s      %6.2f s\n",
	  difftime (end_wall, start_wall),
	  (double) cpu_time / CLOCKS_PER_SEC,
	  (double) cudaTime / 1000.0);

  /* free allocated memory on the GPU					*/
  ret = cudaFree (dev_x);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (dev_y);
  CheckRetValueOfCudaFunction (ret);

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  return EXIT_SUCCESS;
}


/* Simplified implementation of the DAXPY subprogram (double
 * precision alpha x plus y) from the Basic Linear Algebra
 * Subprogram library (BLAS). This subprogram computes
 * "y = alpha * x + y" with identical increments of size "1"
 * for the indexes of both vectors, so that we can omit the
 * increment parameters in the original function which has
 * the following prototype.
 *
 * void daxpy (int n, double alpha, double x[], int incx,
 *	       double y[], int incy);
 *
 *
 * input parameters:	n	number of elements in x and y
 *			alpha	scalar alpha for multiplication
 *			x	elements of vector x
 *			y	elements of vector y
 * output parameters:	y	updated elements of vector y
 * return value:	none
 * side effects:	elements of vector y will be overwritten
 *			  with new values
 *
 */
__global__ void daxpy (const int n, const double alpha,
		       const double * __restrict__ x,
		       double * __restrict__ y)
{
  for (int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
       tid < n;
       tid += blockDim.x * gridDim.x)
  {
    y[tid] += (alpha * x[tid]);
  }
}
