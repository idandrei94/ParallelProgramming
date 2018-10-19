/* A small OpenMP program which measures the time to initialze a
 * vector. This version initializes the vector with a constant.
 * Vectors with more than 1.000.000 elements may profit from a
 * parallel initialization.
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
 *    -o omp_parallel_for_1_with_stubs omp_parallel_for_1_with_stubs.c \
 *    [omp_stubs.c] [-lrt] -lm
 *
 * gcc [-DCygwin] [-DDarwin] [-DLinux] [-fopenmp] \
 *     -o omp_parallel_for_1_with_stubs omp_parallel_for_1_with_stubs.c \
 *     [omp_stubs.c] [-lrt] -lm
 *
 * icc [-DLinux] [-qopenmp] \
 *     -o omp_parallel_for_1_with_stubs omp_parallel_for_1_with_stubs.c \
 *     [omp_stubs.c] [-lrt] -lm
 *
 * cl [/DWin32] /GL /Ox [/openmp] omp_parallel_for_1_with_stubs.c \
 *    [omp_stubs.c]
 *
 *
 * Running:
 *   ./omp_parallel_for_1_with_stubs [size of vector]
 *
 *
 * File: omp_parallel_for_1_with_stubs.c	Author: S. Gross
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

#define DEF_VECTOR_SIZE 1000000		/* default vector size		*/

int main (int argc, char *argv[])
{
  int    size,				/* current vector size		*/
	 i;				/* loop variable		*/
  double *a;				/* vector a[size]		*/
  double wall_clock_time,
	 clock_tick;

  switch (argc)
  {
    case 1:				/* no parameters on cmd line	*/
      size = DEF_VECTOR_SIZE;
      break;

    case 2:				/* one parameter on cmd line	*/
      size = atoi (argv[1]);
      if (size < 1)
      {
	fprintf (stderr, "\n\nError: Vector size must be greater "
		 "than zero.\n"
		 "I use the default size.\n");
	size = DEF_VECTOR_SIZE;
      }
      break;

    default:
      fprintf (stderr, "\n\nError: too many parameters.\n"
	       "Usage: %s [size of vector]\n", argv[0]);
      exit (EXIT_FAILURE);
  }
  /* allocate memory for vector						*/
  a = (double *) malloc (size * sizeof (double));
  if (a == NULL)
  {
    fprintf (stderr, "File: %s, line %d: Can't allocate memory.\n",
	     __FILE__, __LINE__);
    exit (EXIT_FAILURE);
  }

  wall_clock_time = omp_get_wtime ();
  #pragma omp parallel for
  for (i = 0; i < size; ++i)
  {
    a[i] = 0.0;
  }
  wall_clock_time = omp_get_wtime () - wall_clock_time;
  clock_tick      = omp_get_wtick ();
  printf ("a[%d] = %g\n"
	  "elapsed time:   %.9f seconds\n"
	  "time precision: %.9f seconds\n",
	  0, a[0], wall_clock_time, clock_tick);
  free (a);
  return EXIT_SUCCESS;
}
