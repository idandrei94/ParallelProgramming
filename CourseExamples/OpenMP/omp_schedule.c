/* A small OpenMP program which can be used to show different
 * scheduling behaviour.
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
 * cc  -xopenmp -o omp_schedule omp_schedule.c
 * gcc -fopenmp -o omp_schedule omp_schedule.c
 * icc -qopenmp -o omp_schedule omp_schedule.c
 * cl  /GL /Ox /openmp omp_schedule.c
 *
 *
 * Running:
 *   setenv OMP_SCHEDULE static,10
 *   ./omp_schedule
 *
 *
 * File: omp_schedule.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main (void)
{
  int i;				/* loop variable		*/

  #pragma omp parallel for default(none) private(i) schedule(runtime)
  for (i = 0; i < 20; ++i)
  {
    printf ("Thread %d work on loop index %d\n",
	    omp_get_thread_num (), i);
  }
  return EXIT_SUCCESS;
}
