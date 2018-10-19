/* Compute the dot product of two vectors in parallel on an
 * accelerator (GPU) with OpenMP.
 *
 * Compiling:
 *   pgcc -acc -ta=nvidia -fast -Minfo=all \
 *	  -o dot_prod_OpenACC dot_prod_OpenACC.c
 *   gcc  -fopenacc [-foffload=nvptx-none] \
 *	  -o dot_prod_OpenACC dot_prod_OpenACC.c
 *
 * Running:
 *   ./dot_prod_OpenACC
 *
 *
 * File: dot_prod_OpenACC.c		Author: S. Gross
 * Date: 03.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENACC
  #include <openacc.h>
#endif

#define VECTOR_SIZE 100000000		/* vector size (10^8)		*/

/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double a[VECTOR_SIZE],		/* vectors for dot product	*/
	      b[VECTOR_SIZE];


int main (void)
{
  double sum;

  /* initialize vectors							*/
  #pragma acc parallel loop independent \
    copyout(a[0:VECTOR_SIZE], b[0:VECTOR_SIZE])
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    a[i] = 2.0;
    b[i] = 3.0;
  }

  #ifdef _OPENACC
    printf ("Supported standard:                _OPENACC = %d\n"
	    "Number of host devices:            %d\n"
	    "Number of none host devices:       %d\n"
	    "Number of attached NVIDIA devices: %d\n",
	    _OPENACC,
	    acc_get_num_devices(acc_device_host),
	    acc_get_num_devices(acc_device_not_host),
	    acc_get_num_devices(acc_device_nvidia));
  #endif

  /* compute dot product						*/
  sum = 0.0;
  #pragma acc data copyin(a[0:VECTOR_SIZE], b[0:VECTOR_SIZE]) copy(sum)
  #pragma acc parallel loop independent reduction(+:sum)
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    sum += a[i] * b[i];
  }
  printf ("sum = %e\n", sum);
  return EXIT_SUCCESS;
}
