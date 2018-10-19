/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version uses "dgemm" from the "GNU Scientific
 * Library" (http://www.gnu.org/software/gsl) which is based on
 * "dgemm" from BLAS (Basic Linear Algebra Subprograms,
 * http://www.netlib.org/blas). "DGEMM" is an abbreviation for "Double
 * GEneral Matrix Multiplication". DGEMM supports the more general
 * multiplication "c = alpha * op(a) * op(b) + beta * c" where op(.)
 * defines the form of the matrix (normal or transposed).
 *
 * The "GNU Scientific Library" uses a special data structure to
 * represent matrices, so that you must provide a "view" for all
 * matrices.
 *
 * int gsl_blas_dgemm (CBLAS_TRANSPOSE_t TransA,
 *		       CBLAS_TRANSPOSE_t TransB,
 *		       double alpha, const gsl_matrix *A,
 *				     const gsl_matrix *B,
 *		       double beta,	   gsl_matrix *C)
 *
 * TransA, TransB: CblasNoTrans -> op(x) = x
 *		   CblasTrans   -> op(x) = x transposed
 * alpha:	   scalar value
 * A:		   address of matrix "a"
 * B:		   address of matrix "b"
 * beta:	   scalar value
 * C:		   address of matrix "c"
 *
 * 
 * Compiling:
 *   The "GNU Scientific Library" uses single capital letters as
 *   variable names (among others "P") so that you must use "MY_P"
 *   instead of "P" to avoid a clash.
 *
 *   gcc [-DMY_P=<value>] [-DQ=<value>] [-DR=<value>] \
 *	 -o <program name> <source code file name> \
 *	 -lgsl -lgslcblas -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_dgemm_gsl.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <gsl/gsl_blas.h>

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
  double  tmp;				/* temporary value		*/
  time_t  st_mult_ab, et_mult_ab;	/* start/end time (wall clock)	*/
  clock_t ct_mult_ab;			/* used cpu time		*/
  gsl_matrix_view gslMatrix_A,		/* necessary for "dgemm"	*/
		  gslMatrix_B,
		  gslMatrix_C;

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

  /* initialize matrix "c"						*/
  for (i = 0; i < MY_P; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      c[i][j] = 0.0;
    }
  }

  /* create "view"							*/
  gslMatrix_A = gsl_matrix_view_array ((double *) a, MY_P, Q);
  gslMatrix_B = gsl_matrix_view_array ((double *) b, Q, R);
  gslMatrix_C = gsl_matrix_view_array ((double *) c, MY_P, R);
  /* compute result matrix "c" and measure some times			*/
  st_mult_ab = time (NULL);
  ct_mult_ab = clock ();
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &gslMatrix_A.matrix,
		  &gslMatrix_B.matrix, 0.0, &gslMatrix_C.matrix);
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
