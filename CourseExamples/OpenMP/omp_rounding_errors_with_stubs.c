/* A small OpenMP program which shows that rounding errors may
 * influence a result, if different numbers of threads are used.
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
 *    -o omp_rounding_errors_with_stubs \
 *    omp_rounding_errors_with_stubs.c [omp_stubs.c] [-lrt] -lm
 *
 * gcc [-DCygwin] [-DDarwin] [-DLinux] [-fopenmp] \
 *     -o omp_rounding_errors_with_stubs \
 *     omp_rounding_errors_with_stubs.c [omp_stubs.c] [-lrt] -lm
 *
 * icc [-DLinux] [-qopenmp] -o omp_rounding_errors_with_stubs \
 *     omp_rounding_errors_with_stubs.c [omp_stubs.c] [-lrt] -lm
 *
 * cl [/DWin32] /GL /Ox [/openmp] omp_rounding_errors_with_stubs.c \
 *    [omp_stubs.c]
 *
 *
 * Running:
 *   setenv OMP_SCHEDULE static,10
 *   ./omp_rounding_errors_with_stubs [number_of_iterations]
 *
 *
 * File: omp_rounding_errors_with_stubs.c	Author: S. Gross
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

#define NUM_ITERATIONS	50000000	/* 50.000.000			*/

char *scheduling[] = {"unknown", "static", "dynamic", "guided", "auto"};

int main (int argc, char *argv[])
{
  double result = 0.0;
  int	 last,				/* last iteration		*/
	 i;				/* loop variable		*/

  if (argc == 1)
  {
    last = NUM_ITERATIONS;		/* default number of iterations	*/
  }
  else
  {
    last = atoi (argv[1]);		/* use user provided number	*/
    if ((last < 0))
    {
      last = -last;			/* convert to positive number	*/
    }
  }

  #pragma omp parallel
  {
    #pragma omp for reduction(+:result) schedule(runtime)
    for (i = 0; i < last; ++i)
    {
      if (i == 0)
      {
	#if (_OPENMP > 200505)
	  int modifier;			/* param for omp_get_schedule()	*/
	  omp_sched_t kind;		/* param for omp_get_schedule()	*/

	  modifier = 0;
	  kind     = (omp_sched_t) 0;
	  omp_get_schedule (&kind, &modifier);
	  if ((kind > (sizeof (scheduling) / sizeof (scheduling[0]))) ||
	      (kind < 0))
	  {
	    kind = (omp_sched_t) 0;
	  }
	  printf ("number of threads: %d\n"
		  "Scheduling: %s,%d\n",
		  omp_get_num_threads(), scheduling[kind], modifier);
	#else
	  printf ("number of threads: %d\n", omp_get_num_threads());
        #endif
      }
      result += ((double) 1 / 3);
    }
  }
  printf ("Number of iterations: %d\n"
	  "1/3 = %.9f\n"
	  "result = %.9f should be %.9f\n",
	  last, ((double) 1 / 3), result, last * ((double) 1 / 3));
  return EXIT_SUCCESS;
}
