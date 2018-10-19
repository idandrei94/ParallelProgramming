/* This program computes the cache line mapping for a matrix
 * multiplication c = a * b with "double a[P][Q], b[Q][R], c[P][R]".
 * The algorithm uses a row by row multiplication schema which
 * fits better with cachelines and the way how matrices are stored in
 * the programming language "C". At first it multiplies the first
 * element of the first row of "a" with the elements of the first row
 * of "b" to compute the first partial sums of the elements of the
 * first row of "c". Next it multiplies the second element of the
 * first row of "a" with the elements of the second row of "b" to
 * compute the second partial sums of the elements of the first row
 * of "c", and so on until all partial sums for the elements of the
 * first row of "c" are computed. Then it continues the same process
 * with the second row of "a" to compute the second row of "c", and
 * so on until all elements of "c" are computed.
 * The program uses virtual addresses so that its results may differ
 * from real computations where the virtual addresses of the matrix
 * elements will be mapped into physical memory and the cache works
 * with physical addresses. To make things more complicated some
 * operating systems (e.g., Solaris) use page coloring to improve the
 * cache hit ratio. One more issue is that this simulation uses a
 * simple Round-Robin assignment of cache lines in case of a miss and
 * therefore has probably once more nothing to do with reality which
 * normally uses a variation of a Least-Recently-Used or
 * Least-Frequently-Used algorithm. Nevertheless it may be helpful
 * to know the cache line mapping for different matrices (even then,
 * when it is only for virtual addresses and an incorrect cache line
 * replacement algorithm).
 *
 * The program computes the cache lines for n-way caches. If you
 * choose "n = 1" you use a direct-mapped cache and if you choose
 * "n = cache_size / cache_line_size" you use a fully-associative
 * cache.
 *
 *
 * Compiling:
 *   gcc [-DP=<value>] [-DQ=<value>] [-DR=<value>] \
 *	 [-DCSIZE=<value>] [-DCLINE=<value>] [-DNWAYS=<value>] \
 *	 [-DPSIZE=<value>] [-DSET_HITS_MISSES] [-DFIRST_COLUMN] \
 *	 -o <program name> <source code file name> -lm
 *
 *   Example: You can use the following command for matrices with
 *	      512 rows and columns, a 2-way set associative cache
 *	      with a size of 64 KB, a cache line size of 64 bytes,
 *	      4 KB pages, 48x48 submatrices, a table with all hits
 *	      and misses for all cache sets, and information only
 *	      for the computation of the first column of the result
 *	      matrix.
 *
 *	      gcc -DP=512 -DQ=512 -DR=512 -D"CSIZE=(64*1024)" \
 *		  -DCLINE=64 -DNWAYS=2 -D"PSIZE=(4*1024)" \
 *		  -DSET_HITS_MISSES -DFIRST_COLUMN \
 *		  -o mat_mult_block_cacheline_ikj \
 *		  mat_mult_block_cacheline_ikj.c -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: mat_mult_cacheline_ikj.c	Author: S. Gross
 * Date: 07.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <float.h>

#define EPS	DBL_EPSILON		/* from float.h (2.2...e-16)	*/

#ifndef P
  #define P	1984			/* number of rows		*/
#endif
#ifndef Q
  #define Q	1984			/* number of rows / columns	*/
#endif
#ifndef R
  #define R	1984			/* number of columns		*/
#endif

/* default parameters for L2 cache of Sparc64 VII processor:
 * 10-way set associative, 5 MB size, 256 byte cache line, physically
 * indexed, physically tagged
 */
#ifndef CSIZE
  #define CSIZE	(5 * 1024 * 1024)	/* cache size in bytes		*/
#endif
#ifndef CLINE
  #define CLINE	(4 * 64)		/* cache line size in bytes	*/
#endif
#ifndef NWAYS
  #define NWAYS	10			/* number of ways		*/
#endif
#ifndef PSIZE
  #define PSIZE	(8 * 1024)		/* page size in byte		*/
#endif

#ifdef FIRST_COLUMN
  #define SET_HITS_MISSES		/* print hits/misses for sets	*/
#endif

#define NSETS (CSIZE) / ((NWAYS) * (CLINE))	/* # of sets in cache	*/

#define HIT		1
#define MISS		0
#define FEW_HITS	5

#define ABS(a)	 ((a) < 0 ? -(a) : (a))		/* absolute value	*/
#define MAX(a,b) ((a) < (b) ? (b) : (a))	/* maximum value	*/
#define MIN(a,b) ((a) < (b) ? (a) : (b))	/* minimum value	*/

/* Each cache set contains NWAYS associative cache lines. The following
 * data structure contains the start address of the cache line in
 * virtual memory so that it is easy to determine if the data for a
 * special address will be in the cache. For every cache set the hits
 * and misses for all three matrices will be counted. All associative
 * cache lines of a set will be filled in round-robin order ("cyclic
 * buffer") because the real cache line replacement algorithm is
 * unknown.
 */
struct cache_set
{
  intptr_t	addr_block[(NWAYS)];	/* start addresses		*/
  long long	cache_hit_a, cache_hit_b, cache_hit_c,
		cache_miss_a, cache_miss_b, cache_miss_c;
  int		wr_idx,			/* overwrite addr_block[wr_idx]	*/
		padding;		/* to alignment boundary	*/
};

/* Define the cache as a global data structure so that it can be
 * easily used in functions.
 */
static struct cache_set cache[(NSETS)];

/* check and update cache						*/
void update_cache (int set_num, intptr_t addr,
		   long long *hit, long long *miss);
/* determine set number in cache					*/
int compute_set_num (intptr_t addr);
/* determine if data is already in cache				*/
int hit_or_miss (int set_num, intptr_t addr);


int main (void)
{
  int num_of_pages_a,			/* number of pages for matrices	*/
      num_of_pages_b,
      num_of_pages_c,
      mat_size_a,			/* matrix size in bytes		*/
      mat_size_b,
      mat_size_c,
      mat_cache_lines_a,		/* matrix size in "cache lines"	*/
      mat_cache_lines_b,
      mat_cache_lines_c,
      few_hits_a[FEW_HITS],		/* # cache lines with few hits	*/
      few_hits_b[FEW_HITS],
      few_hits_c[FEW_HITS],
      set_num,				/* set number			*/
      ok,				/* temporary value		*/
      i, j, k;				/* loop variables		*/
  /* memory address of a matrix element					*/
  intptr_t  addr_elem_a, addr_elem_b, addr_elem_c;
  long long cache_hits_a,		/* cache statistics		*/
	    cache_hits_b,
	    cache_hits_c,
	    cache_misses_a, cache_misses_b, cache_misses_c,
	    cache_min_hits_a, cache_min_hits_b, cache_min_hits_c,
	    cache_max_hits_a, cache_max_hits_b, cache_max_hits_c,
	    cache_min_misses_a, cache_min_misses_b, cache_min_misses_c,
	    cache_max_misses_a, cache_max_misses_b, cache_max_misses_c;
  double tmp;				/* temporary value		*/
  /* create matrices in heap memory and not automatically on the stack
   * because the matrix size may be too big for the stack.
   */
  static double a[P][Q], b[Q][R], c[P][R];

  /* test some parameters so that this program will work		*/
  if (((NSETS) * (NWAYS) * (CLINE)) != (CSIZE))
  {
    fprintf (stderr, "Error: wrong cache parameters.\n"
	     "  Number of ways:  %d\n"
	     "  Cache line size: %d bytes\n"
	     "  Number of sets:  %d\n"
	     "  Cache size:      %d bytes\n"
	     "  The product of the first three values must be equal "
	     "to the cache size.\n",
	     NWAYS, CLINE, NSETS, CSIZE);
    exit (EXIT_FAILURE);
  }
  /* NSETS must be a power of two					*/
  set_num = 1;
  do
  {
    if ((NSETS) != set_num)
    {
      set_num += set_num;		/* next power of two		*/
    }
    else
    {
      set_num = 0;			/* NSETS is power of two	*/
    }
  } while ((set_num != 0) && ((NSETS) >= set_num));
  if (set_num != 0)
  {
    fprintf (stderr, "Error: wrong cache parameters.\n"
	     "  Number of sets is not a power of two.\n"
	     "  Number of sets: %d\n", NSETS);
    exit (EXIT_FAILURE);
  }
  /* inizialize values for matrix "a"					*/
  mat_size_a	    = sizeof (a);
  mat_cache_lines_a = mat_size_a / (CLINE);
  if ((mat_size_a % (CLINE)) != 0)
  {
    mat_cache_lines_a++;
  }
  num_of_pages_a = mat_size_a / (PSIZE);
  if ((mat_size_a % (PSIZE)) != 0)
  {
    num_of_pages_a++;
  }
  cache_hits_a       = 0LL;
  cache_misses_a     = 0LL;
  cache_min_hits_a   = INT_MAX;
  cache_max_hits_a   = 0LL;
  cache_min_misses_a = INT_MAX;
  cache_max_misses_a = 0LL;

  /* inizialize values for matrix "b"					*/
  mat_size_b	    = sizeof (b);
  mat_cache_lines_b = mat_size_b / (CLINE);
  if ((mat_size_b % (CLINE)) != 0)
  {
    mat_cache_lines_b++;
  }
  num_of_pages_b = mat_size_b / (PSIZE);
  if ((mat_size_b % (PSIZE)) != 0)
  {
    num_of_pages_b++;
  }
  cache_hits_b       = 0LL;
  cache_misses_b     = 0LL;
  cache_min_hits_b   = INT_MAX;
  cache_max_hits_b   = 0LL;
  cache_min_misses_b = INT_MAX;
  cache_max_misses_b = 0LL;

  /* inizialize values for matrix "c"					*/
  mat_size_c	    = sizeof (c);
  mat_cache_lines_c = mat_size_c / (CLINE);
  if ((mat_size_c % (CLINE)) != 0)
  {
    mat_cache_lines_c++;
  }
  num_of_pages_c = mat_size_c / (PSIZE);
  if ((mat_size_c % (PSIZE)) != 0)
  {
    num_of_pages_c++;
  }
  cache_hits_c       = 0LL;
  cache_misses_c     = 0LL;
  cache_min_hits_c   = INT_MAX;
  cache_max_hits_c   = 0LL;
  cache_min_misses_c = INT_MAX;
  cache_max_misses_c = 0LL;

  /* initialize cache structure						*/
  for (i = 0; i < (NSETS); ++i)
  {
    for (j = 0; j < (NWAYS); ++j)
    {
      cache[i].addr_block[j] = INTPTR_MAX;
    }
    cache[i].cache_hit_a  = 0LL;
    cache[i].cache_hit_b  = 0LL;
    cache[i].cache_hit_c  = 0LL;
    cache[i].cache_miss_a = 0LL;
    cache[i].cache_miss_b = 0LL;
    cache[i].cache_miss_c = 0LL;
    cache[i].wr_idx	  = 0;
  }

  /* initialize few_hits_a, few_hits_b, and few_hits_c			*/
  for (i = 0; i < FEW_HITS; ++i)
  {
    few_hits_a[i] = 0;
    few_hits_b[i] = 0;
    few_hits_c[i] = 0;
  }

  /* Now we can do the real work.
   *
   * initialize matrix "a"
   */
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

  /* initialize matrix "c"						*/
  for (i = 0; i < P; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      c[i][j] = 0.0;
    }
  }

  /* compute result matrix "c" and produce cache statistics,
   * compute and add partial sums row-by-row
   */
  for (i = 0; i < P; ++i)
  {
    for (k = 0; k < Q; ++k)
    {
      #ifndef FIRST_COLUMN
        for (j = 0; j < R; ++j)
      #else
        for (j = 0; j < 1; ++j)
      #endif
      {
	c[i][j] += a[i][k] * b[k][j];
	addr_elem_a = (intptr_t) &a[i][k];
	addr_elem_b = (intptr_t) &b[k][j];
	addr_elem_c = (intptr_t) &c[i][j];
	set_num     = compute_set_num (addr_elem_a);
	update_cache (set_num, addr_elem_a,
		      &cache[set_num].cache_hit_a,
		      &cache[set_num].cache_miss_a);
	set_num     = compute_set_num (addr_elem_b);
	update_cache (set_num, addr_elem_b,
		      &cache[set_num].cache_hit_b,
		      &cache[set_num].cache_miss_b);
	set_num     = compute_set_num (addr_elem_c);
	update_cache (set_num, addr_elem_c,
		      &cache[set_num].cache_hit_c,
		      &cache[set_num].cache_miss_c);
      }
    }
  }
  #ifdef SET_HITS_MISSES
    /* print hits and misses for all cache sets				*/
    for (i = 0; i < (NSETS); ++i)
    {
      printf ("set: %5d    hits: %10lld    misses: %10lld\n",
	      i, cache[i].cache_hit_b, cache[i].cache_miss_b);
    }
  #endif

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

  /* collect cache statistics						*/
  for (i = 0; i < (NSETS); ++i)
  {
    cache_hits_a       += cache[i].cache_hit_a;
    cache_misses_a     += cache[i].cache_miss_a;
    cache_min_hits_a   = MIN(cache_min_hits_a, cache[i].cache_hit_a);
    cache_max_hits_a   = MAX(cache_max_hits_a, cache[i].cache_hit_a);
    cache_min_misses_a = MIN(cache_min_misses_a, cache[i].cache_miss_a);
    cache_max_misses_a = MAX(cache_max_misses_a, cache[i].cache_miss_a);
    cache_hits_b       += cache[i].cache_hit_b;
    cache_misses_b     += cache[i].cache_miss_b;
    cache_min_hits_b   = MIN(cache_min_hits_b, cache[i].cache_hit_b);
    cache_max_hits_b   = MAX(cache_max_hits_b, cache[i].cache_hit_b);
    cache_min_misses_b = MIN(cache_min_misses_b, cache[i].cache_miss_b);
    cache_max_misses_b = MAX(cache_max_misses_b, cache[i].cache_miss_b);
    cache_hits_c       += cache[i].cache_hit_c;
    cache_misses_c     += cache[i].cache_miss_c;
    cache_min_hits_c   = MIN(cache_min_hits_c, cache[i].cache_hit_c);
    cache_max_hits_c   = MAX(cache_max_hits_c, cache[i].cache_hit_c);
    cache_min_misses_c = MIN(cache_min_misses_c, cache[i].cache_miss_c);
    cache_max_misses_c = MAX(cache_max_misses_c, cache[i].cache_miss_c);
  }
  /* Let's see if we have sets with very few hits.			*/
  for (i = 0; i < (NSETS); ++i)
  {
    if (cache[i].cache_hit_a < FEW_HITS)
    {
      few_hits_a[cache[i].cache_hit_a]++;
    }
    if (cache[i].cache_hit_b < FEW_HITS)
    {
      few_hits_b[cache[i].cache_hit_b]++;
    }
    if (cache[i].cache_hit_c < FEW_HITS)
    {
      few_hits_c[cache[i].cache_hit_c]++;
    }
  }

  /* print results							*/
  printf ("\nCache parameters\n"
	  "  Cache size (bytes):      %20d\n"
	  "  Number of ways:          %20d\n"
	  "  Cache line size (bytes): %20d\n"
	  "  Number of sets:          %20d\n\n"
	  "Memory parameters\n"
	  "  Page size (bytes):       %20d\n\n"
	  "Matrix parameters                             a"
	  "                    b                    c\n"
	  "  Number of rows:          %20d %20d %20d\n"
	  "  Number of columns:       %20d %20d %20d\n"
	  "  data size (bytes):       %20ld %20ld %20ld\n"
	  "  matrix size (bytes):     %20d %20d %20d\n"
	  "  number of pages:         %20d %20d %20d\n"
	  "  number of cache lines:   %20d %20d %20d\n\n"
	  "Cache statistics for matrix                   a"
	  "                    b                    c\n"
	  "  number of cache hits:    %20lld %20lld %20lld\n"
	  "  minimum hits/set:        %20lld %20lld %20lld\n"
	  "  maximum hits/set:        %20lld %20lld %20lld\n"
	  "  number of cache misses:  %20lld %20lld %20lld\n"
	  "  minimum misses/set:      %20lld %20lld %20lld\n"
	  "  maximum misses/set:      %20lld %20lld %20lld\n",
	  CSIZE, NWAYS, CLINE, NSETS, PSIZE, P, Q, P, Q, R, R,
	  (long) sizeof (a[0][0]), (long) sizeof (b[0][0]),
	  (long) sizeof (c[0][0]),
	  mat_size_a, mat_size_b, mat_size_c,
	  num_of_pages_a, num_of_pages_b, num_of_pages_c,
	  mat_cache_lines_a, mat_cache_lines_b, mat_cache_lines_c,
	  cache_hits_a, cache_hits_b, cache_hits_c,
	  cache_min_hits_a, cache_min_hits_b, cache_min_hits_c,
	  cache_max_hits_a, cache_max_hits_b, cache_max_hits_c,
	  cache_misses_a, cache_misses_b, cache_misses_c,
	  cache_min_misses_a, cache_min_misses_b, cache_min_misses_c,
	  cache_max_misses_a, cache_max_misses_b, cache_max_misses_c);
  for (i = 0; i < FEW_HITS; ++i)
  {
    printf ("  sets with %2d hit(s):     %20d %20d %20d\n",
	    i, few_hits_a[i], few_hits_b[i], few_hits_c[i]);
  }
  return EXIT_SUCCESS;
}

/* Check if data of "addr" is in cache and update cache structure.
 *
 * input parameter:	set_num	    search in cache set "set_num"
 *			addr	    address
 *			hit	    address of hit count
 *			miss	    address of miss count
 * output parameter:	none
 * return value:	none
 * side effects:	none
 */
void update_cache (int set_num, intptr_t addr,
		   long long *hit, long long *miss)
{
  int hit_miss;				/* HIT: data in cache		*/

  hit_miss = hit_or_miss (set_num, addr);
  if (hit_miss == HIT)
  {
    (*hit)++;
  }
  else
  {
    (*miss)++;
    /* fill cache line							*/
    cache[set_num].addr_block[cache[set_num].wr_idx] =
      (addr / (CLINE)) * (CLINE);
    if ((NWAYS) > 1)
    {
     /* use associative cache lines as a cyclic buffer (round robin)	*/
     cache[set_num].wr_idx = (cache[set_num].wr_idx + 1) % (NWAYS);
    }
  }
}


/* Compute the number of the cache set. Address structure:
 * <tag bits><bits for set selection><bits for byte selection in
 * cache line>
 *
 * input parameter:	addr	address
 * output parameter:	none
 * return value:	set number
 * side effects:	none
 */
int compute_set_num (intptr_t addr)
{
  int set_num;				/* set number			*/

  /* Shift address "addr" to the right so that the set number starts
   * in the least significant bit.
   */
  switch (CLINE)
  {
    case 16:
      set_num = addr >> 4;
      break;

    case 32:
      set_num = addr >> 5;
      break;

    case 64:
      set_num = addr >> 6;
      break;

    case 128:
      set_num = addr >> 7;
      break;

    case 256:
      set_num = (int) (addr >> 8);
      break;

    default:
      fprintf (stderr, "Error: unsupported cache line size.\n");
      exit (EXIT_FAILURE);
  }
  /* "NSETS - 1" sets all necessary 1-bits to mask the set bits
   * in the address.
   */
  set_num &= ((NSETS) - 1);
  return set_num;
}


/* Determine if data is in cache. The address block of the cache
 * contains the starting address of a cache line in virtual memory.
 * If the data of "addr" is stored in cache, then the difference
 * between both addresses must be lower than the cache line size.
 *
 * input parameter:	set_num     search in cache set "set_num"
 *			addr	    address of data in virtual memory
 * output parameter:	none
 * return value:	HIT or MISS
 * side effects:	none
 */
int hit_or_miss (int set_num, intptr_t addr)
{
  int i;				/* loop variable		*/

  for (i = 0; i < (NWAYS); ++i)
  {
    /* search all entries of the associative cache lines of the set	*/
    if (ABS(addr - cache[set_num].addr_block[i]) < (CLINE))
    {
      return HIT;
    }
  }
  return MISS;
}
