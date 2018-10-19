/* Computation of "pi" using numerical integration.
 *
 * This version uses a sequential implementation.
 *
 *
 * Compiling:
 *   cc  -o pi_sequential pi_sequential.c
 *   gcc -o pi_sequential pi_sequential.c
 *   icc -o pi_sequential pi_sequential.c
 *   cl  pi_sequential.c
 *
 * Running:
 *   ./pi_sequential [number of intervals]
 *
 *
 * File: pi_sequential.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define f(x)	(4.0 / (1.0 + (x) * (x)))

#define	PI_25	3.141592653589793238462643	/* 25 digits of pi	*/
#define DEF_NUM_INTERVALS 50000000	/* default number of intervals	*/

int main (int argc, char *argv[])
{
  int	  num_iter,			/* # of subintervals/iterations	*/
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
  for (i = 0; i < num_iter; i++)
  {
    x   = h * (double) i;
    pi += h * f(x + h2);
  }
  end_wall = time (NULL);
  cpu_time = clock () - cpu_time;

  printf ("\nApproximation for Pi using %d intervals: %.16f\n"
	  "Error: %.1e\n", 
	  num_iter, pi, fabs (pi - PI_25));

  /* show computation time						*/
  printf ("elapsed time      cpu time\n"
	  "    %6.2f s      %6.2f s\n",
	  difftime (end_wall, start_wall),
	  (double) cpu_time / CLOCKS_PER_SEC);
  return EXIT_SUCCESS;
}
