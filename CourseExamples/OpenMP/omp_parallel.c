/* A small OpenMP program to display "Hello World" on the screen.
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
 * cc  -xopenmp -o omp_parallel omp_parallel.c
 * gcc -fopenmp -o omp_parallel omp_parallel.c
 * icc -qopenmp -o omp_parallel omp_parallel.c
 * cl  /GL /Ox /openmp omp_parallel.c
 *
 *
 * Running:
 *   ./omp_parallel
 *
 *
 * File: omp_parallel.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int main (void)
{
  #ifdef _OPENMP
    printf ("Supported standard: _OPENMP = %d\n", _OPENMP);
  #endif
  #pragma omp parallel
  {
    #ifdef _OPENMP
      printf ("\"Hello World\" from thread %d of %d.\n",
	      omp_get_thread_num (), omp_get_num_threads ());
    #else
      printf ("Without OpenMP: \"Hello World\".\n");
    #endif
  }
  return EXIT_SUCCESS;
}
