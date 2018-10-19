/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version uses "dgemm" from the "Oracle Solaris Studio
 * Performance Library" which is a well-performing implementation of
 * "dgemm" from BLAS (Basic Linear Algebra Subprograms,
 * http://www.netlib.org/blas). "DGEMM" is an abbreviation for "Double
 * GEneral Matrix Multiplication". "dgemm" from the "Oracle Solaris
 * Studio Performance Library" is already parallelized so that it can
 * take advantage of multiple processors and/or processor cores.
 *
 * If you don't use OpenMP, you can set the environment variable
 * PARALLEL prior to the execution of the program, e.g.,
 * "setenv PARALLEL 8" to create eight parallel threads to perform
 * a parallel matrix multiplication.
 *
 * If you use OpenMP, the serial version of "dgemm" will be used
 * inside parallel regions and outside of parallel regions it will
 * use OMP_NUM_THREADS threads to perform a parallel matrix
 * multiplication.
 *
 * "Oracle Developer Studio" supports a 32-bit and a 64-bit version of
 * DGEMM. In the 32-bit version the type for the size of the matrices
 * is "int" and in the 64-bit version it is "long". DGEMM supports the
 * more general multiplication "c = alpha * op(a) * op(b) + beta * c"
 * where op(.) defines the form of the matrix (normal or transposed).
 *
 * void dgemm (char transa, char transb, int m, int n, int k,
 *	       double alpha, double *a, int lda,
 *			     double *b, int ldb,
 *	       double beta,  double *c, int ldc);
 *
 * void dgemm_64 (char transa, char transb, long m, long n, long k,
 *	          double alpha, double *a, long lda,
 *			        double *b, long ldb,
 *	          double beta,  double *c, long ldc);
 *
 * transa, transb: 'N' -> op(x) = x
 *		   'T' -> op(x) = x transposed
 * m, n, k:	   normal:     a[m][k], b[k][n], c[m][n]
 *		   transposed: a[k][m], b[n][k], c[m][n]
 * alpha:	   scalar value
 * a:		   address of matrix "a"
 * lda:		   first dimension of matrix "a"
 * b:		   address of matrix "b"
 * ldb:		   first dimension of matrix "b"
 * beta:	   scalar value
 * c:		   address of matrix "c"
 * ldc:		   first dimension of matrix "c"
 *
 * 
 * Compiling:
 *   You can only use the C compiler from "Oracle Developer Studio" to
 *   compile this program.
 *
 *   cc [-DMY_P=<value>] [-DQ=<value>] [-DR=<value>] \
 *	-o <program name> <source code file name> -library=sunperf -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_dgemm_sunperf.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sunperf.h>

#define EPS	DBL_EPSILON		/* from float.h (2.2...e-16)	*/

#ifndef MY_P
  #define MY_P	1984			/* # of rows			*/
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
static double a[MY_P][Q], b[Q][R],		/* matrices to multiply		*/
	      c[MY_P][R];			/* result matrix		*/

int main (void)
{
  int	  i, j,				/* loop variables		*/
	  ok;				/* temporary value		*/
  double  tmp,				/* temporary value		*/
	  *const A = a[0],		/* necessary for "dgemm"	*/
	  *const B = b[0],
	  *const C = c[0];
  time_t  st_mult_ab, et_mult_ab;	/* start/end time (wall clock)	*/
  clock_t ct_mult_ab;			/* used cpu time		*/


  /* initialize matrix "a"						*/
  for (i = 0; i < MY_P; ++i)
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
  dgemm ('N', 'N', MY_P, R, Q, 1.0, A, MY_P, B, Q, 0.0, C, MY_P);
  et_mult_ab = time (NULL);
  ct_mult_ab = clock () - ct_mult_ab;

  /* test values of matrix "c"						*/
  tmp = c[0][0];
  ok  = 0;
  for (i = 0; i < MY_P; ++i)
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
	    MY_P, R, MY_P, Q, Q, R);
  }
  else
  {
    printf ("c[%d][%d] = a[%d][%d] * b[%d][%d] was not successful.\n"
	    "%d values differ.\n", MY_P, R, MY_P, Q, Q, R, ok);
  }
  printf ("                      elapsed time      cpu time\n"
	  "Mult \"a\" and \"b\":        %6.2f s      %6.2f s\n",
	  difftime (et_mult_ab, st_mult_ab),
	  (double) ct_mult_ab / CLOCKS_PER_SEC);
  return EXIT_SUCCESS;
}
