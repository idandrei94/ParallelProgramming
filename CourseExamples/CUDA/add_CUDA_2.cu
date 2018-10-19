/* A simple program adding two vectors using a GPU. This version
 * uses Unified Memory.
 *
 *
 * Compiling:
 *   nvcc -arch=sm_50 -o add_CUDA_2 add_CUDA_2.cu
 *   clang --cuda-gpu-arch=sm_50 -o add_CUDA_2 add_CUDA_2.cu -lcudart
 *
 *
 * Running:
 *   ./add_CUDA_2 [vector size] [blocks per grid] [threads per block]
 *
 *
 * File: add_CUDA_2.cu			Author: S. Gross
 * Date: 14.02.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>


#define DEFAULT_VECTOR_SIZE  100000000	/* vector size (10^8)		*/
#define DEFAULT_THREADS_PER_BLOCK 256
#define DEFAULT_BLOCKS_PER_GRID   32


__global__ void vecAdd (const int * __restrict__ a,
			const int * __restrict__ b,
			int * __restrict__ c, const size_t vecSize);
int  checkResult (const int * __restrict__ a,
		  const int * __restrict__ b,
		  const int * __restrict__ c, const size_t vecSize);
void evalCmdLine (int argc, char *argv[], size_t *vecSize,
		  int *blocksPerGrid, int *threadsPerBlock);


/* define macro to check the return value of a CUDA function		*/
#define CheckRetValueOfCudaFunction(val) \
  if (val != cudaSuccess) \
  { \
    fprintf (stderr, "file: %s  line %d: %s.\n", \
	     __FILE__, __LINE__, cudaGetErrorString (val)); \
    cudaDeviceReset (); \
    exit (EXIT_FAILURE); \
  }
    

int main (int argc, char *argv[])
{
  int	  *a, *b, *c,			/* vector addresses		*/
	  result,			/* result of comparison		*/
	  blocksPerGrid,
	  threadsPerBlock;
  size_t  vecSize;			/* vector size			*/
  time_t  start_wall, end_wall,		/* start/end time (wall clock)	*/
	  start_total_wall,
    	  end_total_wall;
  clock_t cpu_time;			/* used cpu time		*/
  cudaError_t ret;			/* CUDA function return value	*/


  /* check for command line argument					*/
  evalCmdLine (argc, argv, &vecSize, &blocksPerGrid, &threadsPerBlock);

  /* measure the total wall clock time					*/
  start_total_wall = time (NULL);

  /* allocate managed memory for all vectors				*/
  printf ("Allocate Managed Memory.\n");
  ret = cudaMallocManaged (&a, vecSize * sizeof (int),
			   cudaMemAttachGlobal);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMallocManaged (&b, vecSize * sizeof (int),
			   cudaMemAttachGlobal);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaMallocManaged (&c, vecSize * sizeof (int),
			   cudaMemAttachGlobal);
  CheckRetValueOfCudaFunction (ret);

  /* initialize vectors on the CPU					*/
  printf ("Initializing arrays.\n");
  for (int i = 0; i < (int) vecSize; ++i)
  {
    a[i] = i;
    b[i] = i;
    c[i] = 0;
  }

  /* add vectors and measure computation time				*/
  printf ("Adding arrays.\n");
  start_wall = time (NULL);
  cpu_time = clock ();
  vecAdd <<<(unsigned int) blocksPerGrid,
	    (unsigned int) threadsPerBlock>>>
    (a, b, c, vecSize);
  /* wait until all kernel tasks have finished				*/
  ret = cudaDeviceSynchronize ();
  CheckRetValueOfCudaFunction (ret);
  cpu_time = clock () - cpu_time;
  end_wall = time (NULL);

  /* check result and clean up						*/
  printf ("Checking result.\n");
  result = checkResult (a, b, c, vecSize);

  printf ("Cleaning up.\n");
  /* free allocated memory						*/
  ret = cudaFree (a);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (b);
  CheckRetValueOfCudaFunction (ret);
  ret = cudaFree (c);
  CheckRetValueOfCudaFunction (ret);
  end_total_wall = time (NULL);

  /* reset current device						*/
  ret = cudaDeviceReset ();
  CheckRetValueOfCudaFunction (ret);

  /* show all times							*/
  printf ("\nelapsed time      cpu time      total wall clock time\n"
	  "    %6.2f s      %6.2f s                   %6.2f s\n",
	  difftime (end_wall, start_wall),
	  (double) cpu_time / CLOCKS_PER_SEC,
	  difftime (end_total_wall, start_total_wall));
  return result;
}


/* Add vectors "a" and "b" and store the result into vector "c"
 * using GPU threads.
 *
 * Input:		a, b		vectors to be added
 *			vecSize		size of all vectors
 * Output		c		result vector
 * Return value:	none
 * Sideeffects:		none
 *
 */
__global__ void vecAdd (const int * __restrict__ a,
			const int * __restrict__ b,
			int * __restrict__ c, const size_t vecSize)
{
  for (int tid = (int) (blockIdx.x * blockDim.x + threadIdx.x);
       tid < (int) vecSize;
       tid += blockDim.x * gridDim.x)
  {
    c[tid] = a[tid] + b[tid];
  }
}


/* Compare the sum of "a" and "b" with the values of "c".
 *
 * Input:		a, b		vectors to be added
 *			c		original result vector
 *			vecSize		size of all vectors
 * Output		none
 * Return value:	EXIT_SUCCESS	if c == a + b
 *			EXIT_FAILURE	if c != a + b
 * Sideeffects:		none
 *
 */
int  checkResult (const int * __restrict__ a,
		  const int * __restrict__ b,
		  const int * __restrict__ c, const size_t vecSize)
{
  int result = EXIT_SUCCESS;

  for (int i = 0; (i < (int) vecSize) && (result == EXIT_SUCCESS); ++i)
  {
    if (c[i] != a[i] + b[i])
    {
      result = EXIT_FAILURE;
    }	
  }
  if (result == EXIT_SUCCESS)
  {
    printf ("Adding two vectors completed successfully.\n");
  }
  else
  {
    printf ("Adding two vectors failed.\n");
  }

  return result;
}


/* Evaluate command line arguments and set the vector size to a
 * default value or a value requested on the command line.
 *
 * Input:		argc		argument count
 *			argv		argument vector
 * Output		vecSize		vector size
 *			blocksPerGrid	
 *			threadsPerBlock	
 * Return value:	none
 * Sideeffects:		terminates the program after printing a
 *			help message, if the command line contains
 *			too many arguments
 *
 */
void evalCmdLine (int argc, char *argv[],
		  size_t *vecSize, int *blocksPerGrid, int *threadsPerBlock)
{
  switch (argc)
  {
    case 1:				/* no parameters on cmd line	*/
      *vecSize	       = DEFAULT_VECTOR_SIZE;
      *blocksPerGrid   = DEFAULT_BLOCKS_PER_GRID;
      *threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
      break;

    case 2:				/* one parameter on cmd line	*/
      *vecSize	       = (size_t) atoi (argv[1]);
      *blocksPerGrid   = DEFAULT_BLOCKS_PER_GRID;
      *threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
      break;

    case 3:				/* two parameters on cmd line	*/
      *vecSize	       = (size_t) atoi (argv[1]);
      *blocksPerGrid   = atoi (argv[2]);
      *threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
      break;
 
    case 4:				/* three parameters on cmd line	*/
      *vecSize	       = (size_t) atoi (argv[1]);
      *blocksPerGrid   = atoi (argv[2]);
      *threadsPerBlock = atoi (argv[3]);
      break;

    default:
      fprintf (stderr, "\nError: too many parameters.\n"
	       "Usage: %s [vector size] [blocks per grid] "
	       "[threads per block]\n"
	       "with \"1 <= blocks per grid <= 65.535\" (compute "
	       "capability <= 2.0),\n"
	       "or   \"1 <= blocks per grid <= 2.147.483.647\" "
	       "(compute capability > 2.0),\n"
	       "and  \"1 <= threads per block <= 1024\"\n\n",
	       argv[0]);
      exit (EXIT_FAILURE);
  }

  /* ensure that all values are valid					*/
  if (*vecSize < 1)
  {
    fprintf (stderr, "\nError: Vector size must be greater than zero.\n"
	     "I use the default size: %d.\n\n", DEFAULT_VECTOR_SIZE);
    *vecSize = DEFAULT_VECTOR_SIZE;
  }
  if (*blocksPerGrid < 1)
  {
    fprintf (stderr, "\nError: Number of blocks per grid must be "
	     "greater than zero.\n"
	     "Use \"1 <= blocks per grid <= 65535\" (compute "
	     "capability <= 2.0)\n"
	     "or  \"1 <= blocks per grid <= 2.147.483.647\" "
	     "(compute capability > 2.0).\n"
	     "I use the default number of blocks per grid: %d.\n\n",
	     DEFAULT_BLOCKS_PER_GRID);
    *blocksPerGrid = DEFAULT_BLOCKS_PER_GRID;
  }
  if ((*threadsPerBlock < 1) || (*threadsPerBlock > 1024))
  {
    fprintf (stderr, "\nError: Wrong number of threads per block.\n"
	     "Use \"1 <= threads per block <= 1024\".\n"
	     "I use the default number of threads per block: %d.\n\n",
	     DEFAULT_THREADS_PER_BLOCK);
    *threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
  }
  fflush (stderr);
}
