/* This program adds the first n integer numbers in parallel using
 * a reduction cluase.
 *
 * The default number of parallel threads depends on the
 * implementation, e.g., just one thread or one thread for every
 * virtual processor. You can for example request four threads, if
 * you set the environment variable "OMP_NUM_THREADS" to "4" before
 * you run the program, e.g., "setenv OMP_NUM_THREADS 4". If you
 * compile the program with the Oracle C compiler (former Sun C
 * compiler) the number of threads is reduced to the number of
 * virtual processors by default if the number of requested threads
 * is greater than the number of virtual processors. You can change
 * this behaviour if you set the environment variable "OMP_DYNAMIC"
 * to "FALSE" before you run the program, e.g., "setenv OMP_DYNAMIC
 * FALSE".
 *
 *
 * Compiling:
 *
 * cc  -xopenmp -o omp_reduction omp_reduction.c
 * gcc -fopenmp -o omp_reduction omp_reduction.c
 * icc -qopenmp -o omp_reduction omp_reduction.c
 * cl  /GL /Ox /openmp omp_reduction.c
 *
 *
 * Running:
 *   ./omp_reduction
 *
 *
 * File: omp_reduction.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#define LAST	1000

int main (void)
{
  int i,				/* loop variable		*/
      sum;				/* sum of some integers		*/

  sum = 0;
  #pragma omp parallel for reduction (+:sum) schedule(static)
  for (i = 0; i <= LAST; ++i)
  {
    sum += i;
  }
  printf ("Sum of the first %d integers: %d\n", LAST, sum);
  return EXIT_SUCCESS;
}
