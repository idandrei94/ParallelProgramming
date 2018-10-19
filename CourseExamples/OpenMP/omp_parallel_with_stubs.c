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
 *   One of the following values must be defined on the command line
 *   to get the correct time in "omp_stubs.c", if you don't provide the
 *   option for an OpenMP program. Furthermore "omp_stubs.c" must be
 *   present and "-lrt" must be specified on SunOS and Linux.
 *     -DCygwin	compile for Cygwin
 *     -DDarwin	compile for Darwin (Apple Mac OS X)
 *     -DLinux	compile for Linux
 *     -DWin32	compile for Microsoft Windows (WIN-32 API)
 *
 * cc [-DCygwin] [-DDarwin] [-DLinux] [-xopenmp] \
 *    -o omp_parallel_with_stubs omp_parallel_with_stubs.c \
 *    [omp_stubs.c] [-lrt] -lm
 *
 * gcc [-DCygwin] [-DDarwin] [-DLinux] [-fopenmp] \
 *     -o omp_parallel_with_stubs omp_parallel_with_stubs.c \
 *     [omp_stubs.c] [-lrt] -lm
 *
 * icc [-DLinux] [-qopenmp] \
 *     -o omp_parallel_with_stubs omp_parallel_with_stubs.c \
 *     [omp_stubs.c] [-lrt]
 *
 * cl [/DWin32] /GL /Ox [/openmp] omp_parallel_with_stubs.c \
 *    [omp_stubs.c]
 *
 *
 * Running:
 *   ./omp_parallel_with_stubs
 *
 *
 * File: omp_parallel_with_stubs.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#else
  #include "omp_stubs.h"
#endif

int main (void)
{
  #ifdef _OPENMP
    printf ("Supported standard: _OPENMP = %d\n", _OPENMP);
  #endif
  #pragma omp parallel
  {
    printf ("\"Hello World\" from thread %d of %d.\n",
	    omp_get_thread_num (), omp_get_num_threads ());
  }
  return EXIT_SUCCESS;
}
