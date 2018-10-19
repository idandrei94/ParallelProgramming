/* Compute the dot product of two vectors in parallel with OpenMP.
 * Every thread works on a block (chunk) of the index space.
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads.
 *
 *
 * Compiling:
 *   gcc -fopenmp -o <program name> <source code file name> -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: dot_prod_block_OpenMP.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#define VECTOR_SIZE 100000000		/* vector size (10^8)		*/

/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double a[VECTOR_SIZE],		/* vectors for dot product	*/
	      b[VECTOR_SIZE];


int main (void)
{
  int i;				/* loop variable		*/
  double sum;

  /* initialize vectors							*/
  #pragma omp parallel for default(none) private(i) shared(a, b)
  for (i = 0; i < VECTOR_SIZE; ++i)
  {
    a[i] = 2.0;
    b[i] = 3.0;
  }

  /* compute dot product						*/
  sum = 0.0;
  #pragma omp parallel for default(none) private(i) shared(a, b) \
    reduction(+:sum)
  for (i = 0; i < VECTOR_SIZE; ++i)
  {
    #if defined _OPENMP && (VECTOR_SIZE < 20)
      printf ("Thread %d: i = %d.\n", omp_get_thread_num (), i);
    #endif
    sum += a[i] * b[i];
  }
  printf ("sum = %e\n", sum);
  return EXIT_SUCCESS;
}
