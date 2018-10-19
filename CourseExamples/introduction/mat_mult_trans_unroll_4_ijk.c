/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version transposes matrix b so that the original
 * columns can be used as rows to profit from the cache structure and
 * improve performance. Furthermore we try to gain some profit from
 * the floating-point pipeline by manually unrolling floating-point
 * operations.
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
 * File: mat_mult_trans_unroll_8_ijk.c	Author: S. Gross
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
	      bt[R][Q],			/* b transposed			*/
	      c[P][R];			/* result matrix		*/

int main (void)
{
  int	  i, j, k,			/* loop variables		*/
	  ok;				/* temporary value		*/
  double  tmp;				/* temporary value		*/
  time_t  st_mult_ab, et_mult_ab;	/* start/end time (wall clock)	*/
  clock_t ct_mult_ab;			/* used cpu time		*/

  /* Q should be a multiple of 4 to keep unrolling simple		*/
  if ((Q % 4) != 0)
  {
    fprintf (stderr, "\"Q = %d\" is not a multiple of 4. Please "
	     "choose for Q\n"
	     "an appropriate value because this program cannot handle\n"
	     "arbitrary values for Q.\n", Q);
    exit (EXIT_FAILURE);
  }
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
  for (i = 0; i < R; ++i)		/* transpose matrix b		*/
  {
    for (j = 0; j < Q; ++j)
    {
      bt[i][j] = b[j][i];
    }
  }
  for (i = 0; i < P; ++i)		/* compute matrix c		*/
  {
    for (j = 0; j < R; ++j)
    {
      c[i][j] = 0.0;
      /* unroll loop to gain profit from floating-point pipeline	*/
      for (k = 0; k < Q; k += 4)
      {
	c[i][j] += ((a[i][k] * bt[j][k]) + (a[i][k+1] * bt[j][k+1]) +
		    (a[i][k+2] * bt[j][k+2]) + (a[i][k+3] * bt[j][k+3]));
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
