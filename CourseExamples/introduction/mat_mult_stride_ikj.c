/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version uses a stride of rows for matrix b so that
 * it can profit from cache hits reducing the overall execution time
 * (if the stride is small enough so that all column elements fit
 * into the cache). Furthermore it uses the "traditional" way
 * multiplying and adding the elements of one row and one column
 * within each stride to build partial sums for the result matrix "c"
 * until the final values of c are determined.
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads, if you compile the program with automatic
 * parallelization.
 *
 * 
 * Compiling:
 *   gcc [-DP=<value>] [-DQ=<value>] [-DR=<value>] \
 *	 [-DSTRIDESIZE=<value>]\
 *	 [-floop-parallelize-all] [-ftree-parallelize-loops=4] \
 *	 -o <program name> <source code file name> -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_stride_ikj.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define EPS	DBL_EPSILON		/* from float.h (2.2...e-16)	*/

/* Current processors contain a cache hierarchy with L1, L2, and
 * sometimes even L3 caches with different organizations, cache line
 * sizes, and capacities. You have 512 "sets" of two cache lines, if
 * you have for example a 2-way cache with a capacity of 64 KB and a
 * cache line size of 64 bytes (e.g., L1 data cache of a Sparc64-VII
 * processor). A read operation of one element of a matrix will
 * always read a whole cache line into the cache. If we assume that
 * our matrices are aligned to cache line boundaries, an access to,
 * for example, any of the elements a[0][0] to a[0][7] will read
 * these eight elements into the cache. If we multiply two matrices,
 * we multiply each row element with a corresponding column element
 * and add the products to get one value of the result matrix. In our
 * example we always read eight elements of a cache line when we read
 * one column element. We will not use any of the additional seven
 * elements if the number of column elements is larger than the
 * number of cache lines of the cache because in that case the cache
 * line will be overwritten with new values before we are able to use
 * the unused old elements, thus leading to cache misses for every
 * column element. We can improve the matrix multiplication if we use
 * all elements of a cache line before we overwrite it with new
 * values. This can be done if we split matrix b into row strides
 * so that all column elements of a stride fit into the cache.
 *
 */
#ifndef P
  #define P		1984		/* # of rows			*/
#endif
#ifndef Q
  #define Q		1984		/* # of columns / rows		*/
#endif
#ifndef R
  #define R		1984		/* # of columns			*/
#endif
#ifndef STRIDESIZE
  #define STRIDESIZE	48		/* # rows in a stride		*/
#endif

/* determine the minimum of two values					*/
#define MIN(a,b)	((a) < (b) ? (a) : (b))

/* matrices of this size are too large for a normal stack size and
 * must be allocated globally or with the keyword "static".
 */
static double a[P][Q], b[Q][R],		/* matrices to multiply		*/
	      c[P][R];			/* result matrix		*/

int main (void)
{
  int	  i, j, k, kb,			/* loop variables		*/
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
  for (i = 0; i < P; ++i)
  {
    /* split column/row in STRIDESIZE blocks				*/
    for (kb = 0; kb < Q; kb += STRIDESIZE)
    {
      for (k = kb; k < MIN ((kb + STRIDESIZE), Q); ++k)
      {
	for (j = 0; j < R; ++j)
	{
	  c[i][j] += a[i][k] * b[k][j];
	}
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
