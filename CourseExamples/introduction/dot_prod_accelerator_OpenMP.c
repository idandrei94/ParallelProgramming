/* Compute the dot product of two vectors in parallel on a CPU or an
 * accelerator (GPU) with OpenMP.
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads.
 *
 *
 * Compiling:
 *  CPU and GPU:
 *   gcc -fopenmp -o dot_prod_accelerator_OpenMP \
 *	dot_prod_accelerator_OpenMP.c
 *   clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
 *	-o dot_prod_accelerator_OpenMP dot_prod_accelerator_OpenMP.c
 *  CPU only:
 *   clang -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu \
 *	-o dot_prod_accelerator_OpenMP dot_prod_accelerator_OpenMP.c
 *
 * Running:
 *   setenv OMP_DEFAULT_DEVICE 0	(CPU)
 *   setenv OMP_DEFAULT_DEVICE 1	(GPU)
 *   dot_prod_accelerator_OpenMP
 *
 *
 * File: dot_prod_accelerator_OpenMP.c	Author: S. Gross
 * Date: 22.08.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#define VECTOR_SIZE 100000000		/* vector size (10^8)		*/

/* heap memory to avoid a segmentation fault due to a stack overflow	*/
static double a[VECTOR_SIZE],		/* vectors for dot product	*/
	      b[VECTOR_SIZE];


int main (void)
{
  double sum;

  /* initialize vectors							*/
  #pragma omp target map (from: a, b)
  #pragma omp parallel for default(none) shared(a, b)
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    a[i] = 2.0;
    b[i] = 3.0;
  }

  #ifdef _OPENMP
    printf ("Number of processors:     %d\n"
	    "Number of devices:        %d\n"
	    "Default device:           %d\n"
	    "Is initial device:        %d\n",
	    omp_get_num_procs (), omp_get_num_devices (),
	    omp_get_default_device (), omp_is_initial_device ());
  #endif

  /* compute dot product						*/
  sum = 0.0;
  #pragma omp target map(to:a,b), map(tofrom:sum)
  #pragma omp parallel for default(none) shared(a, b) reduction(+:sum)
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    sum += a[i] * b[i];
  }
  printf ("sum = %e\n", sum);
  return EXIT_SUCCESS;
}
