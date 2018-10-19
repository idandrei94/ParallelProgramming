/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version uses the "traditional" way multiplying and
 * adding the elements of one row and one column to determine one
 * element of the result matrix.
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads.
 *
 * 
 * Compiling:
 *   gcc -fopenmp [-DP=<value>] [-DQ=<value>] [-DR=<value>] \
 *	 -o <program name> <source code file name> -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_int_OpenMP_ijk.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef P
  #define P	1984			/* # of rows			*/
#endif
#ifndef Q
  #define Q	1984			/* # of columns / rows		*/
#endif
#ifndef R
  #define R	1984			/* # of columns			*/
#endif

/* matrices of this size are too large for a normal stack size and
 * must be allocated globally or with the keyword "static".
 */
static int a[P][Q], b[Q][R],		/* matrices to multiply		*/
	   c[P][R];			/* result matrix		*/

int main (void)
{
  int	  i, j, k,			/* loop variables		*/
	  tmp, ok;			/* temporary values		*/
  time_t  st_mult_ab, et_mult_ab;	/* start/end time (wall clock)	*/
  clock_t ct_mult_ab;			/* used cpu time		*/

  #pragma omp parallel default(none) private(i, j) shared(a, b)
  {
    /* initialize matrix "a"						*/
    #pragma omp for
    for (i = 0; i < P; ++i)
    {
      for (j = 0; j < Q; ++j)
      {
	a[i][j] = 2;
      }
    }

    /* initialize matrix "b"						*/
    #pragma omp for
    for (i = 0; i < Q; ++i)
    {
      for (j = 0; j < R; ++j)
      {
	b[i][j] = 3;
      }
    }
  }

  /* compute result matrix "c" and measure some times			*/
  st_mult_ab = time (NULL);
  ct_mult_ab = clock ();
  #pragma omp parallel for default(none) private(i, j, k) \
    shared(a, b, c)
  for (i = 0; i < P; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      c[i][j] = 0;
      for (k = 0; k < Q; ++k)
      {
	c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  et_mult_ab = time (NULL);
  ct_mult_ab = clock () - ct_mult_ab;

  /* test values of matrix "c"						*/
  tmp = c[0][0];
  ok  = 0;
  #pragma omp parallel for default(none) private(i, j) \
    shared(c, tmp) reduction(+:ok)
  for (i = 0; i < P; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      if (c[i][j] != tmp)
      {
	ok++;
      }
    }
  }

  if (ok == 0)
  {
    printf ("c[%d][%d] = a[%d][%d] * b[%d][%d] was successful.\n",
	    P, R, P, Q, Q, R);
  }
  else
  {
    printf ("c[%d][%d] = a[%d][%d] * b[%d][%d] was not successful.\n"
	    "%d values differ.\n", P, R, P, Q, Q, R, ok);
  }
  printf ("                      elapsed time      cpu time\n"
	  "Mult \"a\" and \"b\":        %6.2f s      %6.2f s\n",
	  difftime (et_mult_ab, st_mult_ab),
	  (double) ct_mult_ab / CLOCKS_PER_SEC);
  return EXIT_SUCCESS;
}
