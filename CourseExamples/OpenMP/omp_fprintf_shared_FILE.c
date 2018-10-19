/* A small OpenMP program which shows different problems and solutions
 * with "stdout", "stderr", and so on, when you define "default(none)".
 *
 * If you use
 *
 * fprintf (stderr, "File: %s, line %d: Can't allocate memory.\n",
 *	    __FILE__, __LINE__);
 *
 * you get an error message similar to
 *
 * xyz.c:72:11: error: '__iob' not specified in enclosing parallel
 *
 *
 * The error message depends on the operating system and the problem
 * must be solved in an operating system specific way, if you want to
 * use the above fprintf-statement.
 *
 * omp_fprintf.c		program with the above statement
 * omp_fprintf_shared_iob.c	solution for SunOS 10, but not for Linux
 * omp_fprintf_shared_stderr.c	solution for Linux, but not for SunOS
 * omp_fprintf_shared_FILE.c	solution for all operating systems
 *
 * Alternatively you can use "perror ()" to avoid compiler specific
 * variables in the enclosing parallel clause and to avoid a special
 * pointer to "stderr". You can also use "sprintf" to build a similar
 * error message as above and use this string as parameter for "perror".
 *
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
 * cc  -xopenmp -o omp_fprintf_shared_FILE omp_fprintf_shared_FILE.c
 * gcc -fopenmp -o omp_fprintf_shared_FILE omp_fprintf_shared_FILE.c
 * icc -qopenmp -o omp_fprintf_shared_FILE omp_fprintf_shared_FILE.c
 * cl  /GL /Ox /openmp omp_fprintf_shared_FILE.c
 *
 *
 * Running:
 *   ./omp_fprintf_shared_FILE
 *
 *
 * File: omp_fprintf_shared_FILE.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main (void)
{
  FILE *fp_stderr = stderr;

  #pragma omp parallel default(none) shared(fp_stderr)
  fprintf (fp_stderr, "Hello!\n");
  return EXIT_SUCCESS;
}
