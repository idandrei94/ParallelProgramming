/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version uses a row by row multiplication schema
 * which fits better with cachelines and the way how matrices are
 * stored in the programming language "C". At first it multiplies the
 * first element of the first row of "a" with the elements of the first
 * row of "b" to compute the first partial sums of the elements of the
 * first row of "c". Next it multiplies the second element of the
 * first row of "a" with the elements of the second row of "b" to
 * compute the second partial sums of the elements of the first row
 * of "c", and so on until all partial sums for the elements of the
 * first row of "c" are computed. Then it continues the same process
 * with the second row of "a" to compute the second row of "c", and
 * so on until all elements of "c" are computed.
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads, if you compile the program with automatic
 * parallelization.
 *
 * 
 * Compiling:
 *   gcc [-DP=<value>] [-DQ=<value>] [-DR=<value>] \
 *	 [-floop-parallelize-all] [-ftree-parallelize-loops=4] \
 *	 -o <program name> <source code file name> -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_ikj.c		       	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define EPS	DBL_EPSILON		/* from float.h (2.2...e-16)	*/

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
static double a[P][Q], b[Q][R],		/* matrices to multiply		*/
	      c[P][R];			/* result matrix		*/

int main (void)
{
  int	  i, j, k,			/* loop variables		*/
	  ok;				/* temporary value		*/
  double  tmp;				/* temporary value		*/
  time_t  st_mult_ab, et_mult_ab;	/* start/end time (wall clock)	*/
  clock_t ct_mult_ab;			/* used cpu time		*/

  /* initialize matrix "a"						*/
  for (i = 0; i < P; ++i)
  {
    for (j = 0; j < Q; ++j)
    {
      a[i][j] = 2.0;
    }
  }

  /* initialize matrix "b"						*/
  for (i = 0; i < Q; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      b[i][j] = 3.0;
    }
  }

  /* compute result matrix "c" and measure some times			*/
  st_mult_ab = time (NULL);
  ct_mult_ab = clock ();
  for (i = 0; i < P; ++i)		/* initialize matrix "c"	*/
  {
    for (j = 0; j < R; ++j)
    {
      c[i][j] = 0.0;
    }
  }
  /* compute and add partial sums row-by-row				*/
  for (i = 0; i < P; ++i)
  {
    for (k = 0; k < Q; ++k)
    {
      for (j = 0; j < R; ++j)
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
  for (i = 0; i < P; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      if (fabs (c[i][j] - tmp) > EPS)
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
