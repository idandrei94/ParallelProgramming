/* Compute the dot product of two vectors sequentially.
 *
 *
 * Compiling:
 *   gcc -o <program name> <source code file name> -lm
 *
 * Running:
 *   <program name>
 *
 *
 * File: dot_prod_sequential.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 100000000		/* vector size (10^8)		*/

/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double a[VECTOR_SIZE],		/* vectors for dot product	*/
	      b[VECTOR_SIZE];


int main (void)
{
  int i;				/* loop variable		*/
  double sum;

  /* initialize vectors							*/
  for (i = 0; i < VECTOR_SIZE; ++i)
  {
    a[i] = 2.0;
    b[i] = 3.0;
  }

  /* compute dot product						*/
  sum = 0.0;
  for (i = 0; i < VECTOR_SIZE; ++i)
  {
    sum += a[i] * b[i];
  }
  printf ("sum = %e\n", sum);
  return EXIT_SUCCESS;
}
