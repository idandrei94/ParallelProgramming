/* A small OpenMP program to display "Hello World" on the screen.
 * This version uses nested parallel sections.
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
 * Generally only one thread will be created for an inner parallel
 * section. You must set the environment variable "OMP_NESTED" to
 * "TRUE" if you want a thread team for the inner parallel region,
 * e.g., "setenv OMP_NESTED TRUE".
 *
 *
 * Compiling:
 *
 * cc  -xopenmp -o omp_parallel_nested omp_parallel_nested.c
 * gcc -fopenmp -o omp_parallel_nested omp_parallel_nested.c
 * icc -qopenmp -o omp_parallel_nested omp_parallel_nested.c
 * cl  /GL /Ox /openmp omp_parallel_nested.c
 *
 *
 * Running:
 *   ./omp_parallel_nested
 *
 *
 * File: omp_parallel_nested.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

/* restrict number of threads on level 1 of nested parallel regions	*/
#define LEVEL_1_THREADS 2

int main (void)
{
  #ifdef _OPENMP
    int threadID;
  #endif

  #ifdef _OPENMP
    printf ("Supported standard: _OPENMP = %d\n", _OPENMP);
  #endif
  #pragma omp parallel private(threadID)
  {
    #ifdef _OPENMP
      threadID = omp_get_thread_num ();
      printf ("Level 0: \"Hello World\" from thread %d of %d.\n",
	      threadID, omp_get_num_threads ());
      /* restrict number of threads if a thread team is allowed in a
       * nested parallel region
       */
      #pragma omp parallel num_threads(LEVEL_1_THREADS)
      {
	printf ("Level 1: My parent thread: %d. \"Bye bye World\" "
		"from thread %d of %d.\n", threadID,
		omp_get_thread_num (), omp_get_num_threads ());
      }
    #else
      printf ("Without OpenMP: \"Hello World\".\n");
    #endif
  }
  return EXIT_SUCCESS;
}
