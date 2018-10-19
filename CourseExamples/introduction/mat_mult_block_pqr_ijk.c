/* The program measures the time to perform the matrix multiplication
 * c = a * b. This version uses arbitrary submatrices so that it can
 * profit from cache hits reducing the overall execution time (if all
 * three current submatrices fit into the cache). Furthermore it uses
 * the "traditional" way multiplying and adding the elements of one
 * row and one column within each submatrix to determine one element
 * of a submatrix of the result matrix "c".
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads, if you compile the program with automatic
 * parallelization.
 *
 * 
 * Compiling:
 *   You can define the matrix sizes and the sizes of the submatrices
 *   on the command line with:
 *   "-DP=<value> -DQ=<value> -DR=<value> -DBLOCKSIZE_P=<value>
 *    -DBLOCKSIZE_Q=<value> -DBLOCKSIZE_R=<value>".
 *
 *         BLOCKSIZE_R            BLOCKSIZE_Q            BLOCKSIZE_R
 *         <---->                 <->                    <---->
 *        ^+-------------+       ^+-------------+       ^+-------------+
 *   BS_P ||    |        |  BS_P ||    |        |  BS_Q ||    |        |
 *        ||    |        |       ||    |        |       v|----+        |
 *        ||    |        |       ||    |        |        |             |
 *        ||    |        |       ||    |        |        |             |
 *        v|----+  c     |    =  v|----+  a     |    *   |       b     |
 *         |             |        |             |        |             |
 *         |             |        |             |        |             |
 *         |             |        |             |        |             |
 *         +-------------+        +-------------+        +-------------+
 *
 *   Matrices "a" and "c" are processed row by row so that the block
 *   size isn't so important. Matrix "b" will be processed column by
 *   column so that BLOCKSIZE_Q should be small and BLOCKSIZE_R large
 *   and a multiple of the cache line size.
 *
 *   gcc [-DP=<value>] [-DQ=<value>] [-DR=<value>] \
 *	 [-DBLOCKSIZE_P=<value>] [-DBLOCKSIZE_Q=<value>] \
 *	 [-DBLOCKSIZE_R=<value>]
 *	 [-floop-parallelize-all] [-ftree-parallelize-loops=4] \
 *	 -o <program name> <source code file name> -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_block_pqr_ijk.c	Author: S. Gross
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
 * values. This can be done if we split our matrices into submatrices
 * which fit into the cache.
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
#ifndef BLOCKSIZE_P
  #define BLOCKSIZE_P	8		/* # rows in submatrix		*/
#endif
#ifndef BLOCKSIZE_Q
  #define BLOCKSIZE_Q	4		/* # rows/columns in submatrix	*/
#endif
#ifndef BLOCKSIZE_R
  #define BLOCKSIZE_R	256		/* # columns in submatrix	*/
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
  int	  i, j, k, ib, jb, kb,		/* loop variables		*/
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
  /* split rows in BLOCKSIZE_P blocks					*/
  for (ib = 0; ib < P; ib += BLOCKSIZE_P)
  {
    /* split columns in BLOCKSIZE_R blocks				*/
    for (jb = 0; jb < R; jb += BLOCKSIZE_R)
    {
      /* split columns/rows in BLOCKSIZE_Q blocks			*/
      for (kb = 0; kb < Q; kb += BLOCKSIZE_Q)
      {
	/* do computations for each submatrix of matrix "c"		*/
	for (i = ib; i < MIN ((ib + BLOCKSIZE_P), P); ++i)
        {
	  for (j = jb; j < MIN ((jb + BLOCKSIZE_R), R); ++j)
          {
	    for (k = kb; k < MIN ((kb + BLOCKSIZE_Q), Q); ++k)
	    {
	      c[i][j] += a[i][k] * b[k][j];
	    }
	  }
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
