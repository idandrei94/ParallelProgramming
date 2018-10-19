/* The statements of each block run sequentially and the blocks run
 * in parallel.
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
 * cc  -xopenmp -o omp_sections omp_sections.c
 * gcc -fopenmp -o omp_sections omp_sections.c
 * icc -qopenmp -o omp_sections omp_sections.c
 * cl  /GL /Ox /openmp omp_sections.c
 *
 *
 * Running:
 *   ./omp_sections
 *
 *
 * File: omp_sections.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#define LAST	10

int main (void)
{
  int i, j, k;				/* loop variables		*/

  #pragma omp parallel sections
  {
    #pragma omp section
    for (i = 0; i < LAST; ++i)
    {
      printf ("First block:  %d\n", i);
    }
    #pragma omp section
    for (j = 0; j < LAST; ++j)
    {
      printf ("Second block: %d\n", j);
    }
    #pragma omp section
    for (k = 0; k < LAST; ++ k)
    {
      printf ("Third block:  %d\n", k);
    }
  }
  return EXIT_SUCCESS;
}
