/* Parallel computation of "Pi" using numerical integration. The
 * program can be compiled for a CPU or parallelized with OpenMP
 * or OpenACC. It uses 4*10^8 intervals as a default so that it
 * takes some time to compute the value.
 *
 * Compiling:
 *   gcc -fopenacc [-foffload=nvptx-none] \
 *	 -o pi_gcc_openacc pi_OpenACC_OpenMP.c
 *   gcc -fopenmp -o pi_gcc_openmp pi_OpenACC_OpenMP.c
 *   gcc -o pi_gcc_cpu pi_OpenACC_OpenMP.c
 *
 *   pgcc -acc -ta=nvidia -Minfo=all \
 *	  -o pi_pgcc_openacc pi_OpenACC_OpenMP.c
 *   pgcc -mp -Minfo=all -o pi_pgcc_openmp pi_OpenACC_OpenMP.c
 *   pgcc -Minfo=all -o pi_pgcc_cpu pi_OpenACC_OpenMP.c
 *
 *
 * Running:
 *   ./pi_gcc_openacc [number of intervals]
 *   ./pi_gcc_openmp [number of intervals]
 *   ./pi_gcc_cpu [number of intervals]
 *
 *   ./pi_pgcc_openacc [number of intervals]
 *   ./pi_pgcc_openmp [number of intervals]
 *   ./pi_pgcc_cpu [number of intervals]
 *
 *   /usr/bin/time -p <one of the above programs>
 *
 *
 * File: pi_OpenACC_OpenMP.c		Author: S. Gross
 * Date: 03.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>

#define NUM_INTERVALS 400000000		/* default number of intervals	*/


#define f(x) (4.0 / (1.0 + (x) * (x)))	/* function to compute "pi"	*/

int main (int argc, char *argv[])
{
  double pi = 0.0,
	 h;				/* width of an interval		*/
  int	 n;				/* number of intervals		*/

  if (argc == 2)
  {
    n = atoi (argv[1]);
  }
  else
  {
    n = NUM_INTERVALS;
  }
  h = 1.0 / (double) n;
  #ifdef _OPENMP
    printf ("Using OpenMP.\n");
    #pragma omp parallel for default(none) shared(h, n) reduction(+:pi)
  #elif _OPENACC
    printf ("Using OpenACC.\n");
    #pragma acc parallel
    #pragma acc loop reduction(+:pi)
  #else
    printf ("Using neither OpenACC nor OpenMP.\n");
  #endif
  for (long i = 0; i < n; i++)
  {
    double x;

    x = (h * (double) i) + (h / 2);	/* midpoint of i-th interval	*/
    pi += h * f(x);			/* tangent-trapezoidal rule	*/
  }
  printf ("pi = %.10f\n", pi);

  return EXIT_SUCCESS;
}
