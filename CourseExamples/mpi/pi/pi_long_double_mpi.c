/* Parallel computation of Pi using numerical integration. The program
 * is based on the example in the book "W. Gropp, et al.: Using MPI -
 * Portable Parallel Programming with the Message-Passing Interface.
 * The MIT Press, 1996, pages 28-30".
 *
 * The program measures the times for computation and communication
 * between the processes. The communication time is the sum of the times
 * for all data transfers  a n d  the waiting time of "task 0" to
 * get the results of the other tasks. Therefore you may get more or
 * less oscillating times, especially if all tasks are running on the
 * same computer (depending on the scheduling) or if "task 0" will be
 * executed on a very fast computer and another task on a heavy-loaded
 * one.
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: pi_long_double_mpi.c		Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define f(x)	(4.0L / (1.0L + (x) * (x)))

#define	PI_25	3.141592653589793238462643L	/* 25 digits		*/

int main (int argc, char *argv[])
{
  int		alg,			/* algorithm			*/
		ntasks,			/* number of parallel tasks	*/
		namelen,		/* length of processor name	*/
		mytid,			/* my task id			*/
		n,			/* number of subintervals      	*/
		err_cnt,		/* input error counter		*/
		more_to_do,
		i;			/* loop variable		*/
  long double	mypi,			/* local sum of pi		*/
		pi,			/* global sum of pi      	*/
		h,			/* length of subinterval       	*/
		x,			/* distinct points xi		*/
		t1, t2, t3, t4;		/* temporary values		*/
  double	scomm_time,		/* starttime for communication	*/
		scomp_time,		/* starttime for computation   	*/
		dcomm_time,		/* duration of communication	*/
		dcomp_time, dcomp_tmp;	/* duration of computation	*/
  char		processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init (&argc, &argv);
  scomm_time = MPI_Wtime ();
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  dcomm_time = MPI_Wtime () - scomm_time;
  MPI_Get_processor_name (processor_name, &namelen);
  /* With the next statement every process executing this code will
   * print one line on the display. It may happen that the lines will
   * get mixed up because the display is a critical section. In general
   * only one process (mostly the process with rank 0) will print on
   * the display and all other processes will send their messages to
   * this process. Nevertheless for debugging purposes (or to
   * demonstrate that it is possible) it may be useful if every
   * process prints itself.
   */
  fprintf (stdout, "Process %d of %d running on %s\n",
	   mytid, ntasks, processor_name);
  fflush (stdout);
  MPI_Barrier (MPI_COMM_WORLD);		/* wait for all other processes	*/

  if (mytid == 0)
  {
    printf ("\n\nThe time will be measured in multiples of "
	    "%.1e seconds.\n", MPI_Wtick ());
    printf ("The initialization phase of MPI took %.3f seconds.\n",
	    dcomm_time);
  }
  more_to_do = 1;
  while (more_to_do == 1)
  {
    if (mytid == 0)
    {
      printf ("\n\nNumerical computation of Pi. Number of intervals: "
	      "(quit: <= 0): ");
      fflush (stdout);
      scanf (" %d", &n);
      if (n > 0)
      {
	err_cnt = 0;
	do
	{
	  printf ("\nThe following algorithms are available for "
		  "the integration:\n"
		  "  1   tangent-trapezoidal-rule\n"
		  "  2   chord-trapezoidal-rule\n"
		  "  3   Simpson-rule\n"
		  "  4   Milne-rule\n"
		  "Please enter the number of the desired rule: ");
	  fflush (stdout);
	  scanf (" %d", &alg);
	  if ((alg < 1) || (alg > 4))
	  {
	    printf ("\n\nError: No admissible value !!!!!!\n\n");
	    err_cnt++;
	  }
	  if (err_cnt > 5)
	  {
	    printf ("Terminating as a result of too many input "
		    "errors!\n\n");
	    exit (EXIT_SUCCESS);
	  }
	} while ((alg < 1) || (alg > 4));
      }
    }
    /* Each task  m u s t  call MPI_Bcast to learn about the algorithm
     * which should be used.
     */
    MPI_Bcast (&alg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /*
     * Now we can start our real work!
     */
    scomm_time = MPI_Wtime ();	      /* 1st part of communication time	*/
    MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    dcomm_time = MPI_Wtime () - scomm_time;
    if (n <= 0)
    {
      more_to_do = 0;			/* terminate			*/
    }
    else
    {
      scomp_time = MPI_Wtime ();
      h    = 1.0L / (long double) n;
      mypi = 0.0L;
      switch (alg)
      {
	case 1:				/* tangent-trapezoidal rule	*/
	  t1 = h / 2;
	  for (i = mytid; i < n; i += ntasks)
	  {
	    x     = h * (long double) i;
	    mypi += h * f(x + t1);
	  }
	  break;

	case 2:				/* chord-trapezoidal rule      	*/
	  t1 = h / 2;
	  for (i = mytid; i < n; i += ntasks)
	  {
	    x	  = h * (long double) i;
	    mypi += t1 * (f(x) + f(x + h));
	  }
	  break;

	case 3:				/* Simpson-rule			*/
	  t1 = h / 2;
	  t2 = h / 6;
	  for (i = mytid; i < n; i += ntasks)
	  {
	    x     = h * (long double) i;
	    mypi += t2 * (f(x) + 4 * f(x + t1) + f(x + h));
	  }
	  break;

	case 4:				/* Milne-rule			*/
	  t1 = h / 4;
	  t2 = h / 2;
	  t3 = 3 * h / 4;
	  t4 = h / 90;
	  for (i = mytid; i < n; i += ntasks)
	  {
	    x     = h * (long double) i;
	    mypi += t4 * (7 * (f(x) + f(x + h)) + 12 * f(x + t2) +
		    32 * (f(x + t1) + f(x + t3)));
	  }
	  break;

	default:
	  if (mytid == 0)
	  {
	    printf ("\nAlgorithm %d isn't implemented yet.\n",
		    alg);
	  }
      }
      dcomp_tmp = MPI_Wtime () - scomp_time;
      /* Because the following MPI_Reduce call is for statistical
       * reasons only, it will not be integrated in the communication
       * time.
       */
      MPI_Reduce (&dcomp_tmp, &dcomp_time, 1, MPI_DOUBLE,
		  MPI_SUM, 0, MPI_COMM_WORLD);
      scomm_time = MPI_Wtime ();      /* 2nd part of communication time	*/
      MPI_Reduce (&mypi, &pi, 1, MPI_LONG_DOUBLE, MPI_SUM, 0,
		  MPI_COMM_WORLD);
      dcomm_time += MPI_Wtime () - scomm_time;
      if (mytid == 0)
      {
	printf ("\nApproximation for Pi using %d intervals: %.16Lf\n"
		"Error:                               %.1e\n", 
		n, pi, fabs ((double) (pi - PI_25)));
	printf ("Duration of the computation:         %.6f Sekunden\n",
		dcomp_time);
	printf ("Duration of communication in task 0: %.6f Sekunden\n",
		dcomm_time);
      }
    }
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
