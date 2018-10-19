/* Computation of "pi" using numerical integration.
 *
 * This version uses OpenMP and the "reduction" clause.
 *
 * The default number of parallel threads depends on the
 * implementation, e.g., just one thread or one thread for every
 * virtual processor. You can for example request four threads, if
 * you set the environment variable "OMP_NUM_THREADS" to "4" before
 * you run the program, e.g., "setenv OMP_NUM_THREADS 4". If you
 * compile the program with the Oracle C compiler (former Sun C
 * compiler) the number of threads is reduced to the number of
 * virtual processors by default if the number of requested threads
 * is greater than the number of virtual processors. You can change
 * this behaviour if you set the environment variable "OMP_DYNAMIC"
 * to "FALSE" before you run the program, e.g., "setenv OMP_DYNAMIC
 * FALSE".
 *
 * Compiling:
 *   cc  -xopenmp -o pi_reduction pi_reduction.c [omp_stubs.c] [-lrt]
 *   gcc -fopenmp -o pi_reduction pi_reduction.c [omp_stubs.c] [-lrt]
 *   icc -qopenmp -o pi_reduction pi_reduction.c [omp_stubs.c]
 *   cl  /GL /Ox /openmp pi_reduction.c [omp_stubs.c]
 *
 * Running:
 *   ./pi_reduction [number of intervals]
 *
 *
 * File: pi_reduction.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
  #include <omp.h>
#else
  #include "omp_stubs.h"
#endif

#define f(x)	(4.0 / (1.0 + (x) * (x)))

#define	PI_25	3.141592653589793238462643	/* 25 digits of pi	*/
#define DEF_NUM_INTERVALS 50000000	/* default number of intervals	*/

int main (int argc, char *argv[])
{
  int	  num_iter,			/* # of subintervals/iterations	*/
	  num_threads,			/* number of parallel threads	*/
	  i;				/* loop variable		*/
  double  pi,				/* computed value of pi      	*/
	  h,				/* length of subinterval       	*/
	  h2,				/* value for h/2		*/
	  x;				/* distinct points xi		*/
  time_t  start_wall, end_wall;		/* start/end time (wall clock)	*/
  clock_t cpu_time;			/* used cpu time		*/

  switch (argc)
  {
    case 1:				/* no parameters on cmd line	*/
      num_iter = DEF_NUM_INTERVALS;
      break;

    case 2:				/* one parameter on cmd line	*/
      num_iter = atoi (argv[1]);
      if (num_iter < 1)
      {
	fprintf (stderr, "\n\nError: Number of intervals must be "
		 "greater than zero.\n"
		 "I use the default size.\n");
	num_iter = DEF_NUM_INTERVALS;
      }
      break;

    default:
      fprintf (stderr, "\n\nError: too many parameters.\n"
	       "Usage: %s [number of intervals]\n", argv[0]);
      exit (EXIT_FAILURE);
  }

  /* compute "pi" with the tangent-trapezoidal rule and measure
   * computation time
   */
  start_wall = time (NULL);
  cpu_time   = clock ();
  pi = 0.0;
  h  = 1.0 / (double) num_iter;
  h2 = h / 2;
  #pragma omp parallel default(none) private(i, x) \
    shared(h, h2, num_iter, num_threads) reduction(+:pi)
  {
    #pragma omp single
    {
      num_threads = omp_get_num_threads ();
    }
    #pragma omp for
    for (i = 0; i < num_iter; i++)
    {
      x   = (h * (double) i) + h2;
      pi += h * f(x);
    }
  }
  end_wall = time (NULL);
  cpu_time = clock () - cpu_time;

  printf ("\nApproximation for Pi using %d intervals and %d threads\n"
	  "  on %d processors: %.16f\n"
	  "Error: %.1e\n", 
	  num_iter, num_threads, omp_get_num_procs (),
	  pi, fabs (pi - PI_25));

  /* show computation time						*/
  printf ("elapsed time      cpu time\n"
	  "    %6.2f s      %6.2f s\n",
	  difftime (end_wall, start_wall),
	  (double) cpu_time / CLOCKS_PER_SEC);
  return EXIT_SUCCESS;
}
