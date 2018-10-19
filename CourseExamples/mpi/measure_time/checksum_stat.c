/* This program determines all numbers in a defined intervall (from
 * 1 up to at most INT_MAX) which satisfy the following condition:
 * The square of the sum of the digits of the number is equal to the
 * sum of digits of the square of the number. Furthermore the program
 * measures the time needed to compute the numbers. This version
 * "distributes" the work equally among all processes (static
 * distribution).
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: checksum_stat.c			Author: S. Gross
 * Date: 16.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "mpi.h"

#define DEF_UP_BOUND	10000000	/* intervall [1, DEF_UP_BOUND]	*/

int checksum (unsigned long long number);


int main (int argc, char *argv[])
{
  int		     mytid,		/* my task id			*/
		     ntasks,		/* number of parallel tasks	*/
		     chk_sum,		/* checksum of digits		*/
		     digits_n,		/* # of digits of "upper_bound"	*/
		     digits_sn,		/* # of digits of squared num	*/
		     namelen;		/* length of processor name	*/
  long		     i,			/* loop variable		*/
		     upper_bound;	/* upper bound of intervall	*/
  unsigned long long sq_num;		/* square of a number		*/
  clock_t	     my_CPUtime,	/* used CPU-time		*/
		     sum_CPUtime;
  double	     elapsed_time;	/* elapsed time on root node	*/
  char		     processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init (&argc, &argv);
  elapsed_time = MPI_Wtime ();
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
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
    /* evaluate command line arguments					*/
    if (argc == 2)
    {
      /* program was called with upper bound of intervall		*/
      upper_bound = atol (argv[1]);
      if ((upper_bound <= 0) || (upper_bound > INT_MAX))
      {
	fprintf (stderr, "\n\nError: Parameter must be between 1 and "
		 "%d\n"
		 "Usage:\n"
		 "  mpiexec -np <number of processes> %s "
		 "[<upper bound>]\n",
		 INT_MAX, argv[0]);
      }
    }
    else
    {
      upper_bound = DEF_UP_BOUND;
    }
  }
  /* Each task  m u s t  call MPI_Bcast to learn the value of
   * "upper_bound"
   */
  MPI_Bcast (&upper_bound, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  if ((upper_bound <= 0) || (upper_bound > INT_MAX))
  {
    /* wrong value							*/
    MPI_Finalize ();
    exit (EXIT_SUCCESS);
  }
  /*
   * Now we can start our real work!
   */
  my_CPUtime   = clock ();
  /*determine how many digits the largest number can have		*/
  digits_n  = (int) (log10 ((double) (upper_bound) + 1));
  digits_sn = (int) (log10 ((double) (upper_bound * upper_bound) + 1));
  for (i = mytid + 1; i <= upper_bound; i += ntasks)
  {
    chk_sum = checksum ((unsigned long long) i);
    sq_num  = (unsigned long long) i *
	      (unsigned long long) i;
    if ((chk_sum * chk_sum) == checksum (sq_num))
    {
      printf ("number: %*ld    number^2: %*lld    "
	      "(checksum (%*ld)) ^ 2: %4d\n",
	      digits_n, i, digits_sn, sq_num, digits_n, i,
	      chk_sum * chk_sum);
    }
  }
  my_CPUtime   = clock () - my_CPUtime;
  elapsed_time = MPI_Wtime () - elapsed_time;
  MPI_Reduce (&my_CPUtime, &sum_CPUtime, 1, MPI_DOUBLE,
	      MPI_SUM, 0, MPI_COMM_WORLD);
  if (mytid == 0)
  {
    printf ("\nelapsed time > (used CPU-time / number of processes)\n"
	    "  elapsed time:        %0.1f seconds\n"
	    "  used CPU-time:       %0.1f seconds\n"
	    "  number of processes: %d\n\n",
	    elapsed_time, (double) sum_CPUtime / CLOCKS_PER_SEC,
	    ntasks);
  }
  MPI_Finalize ();
  return EXIT_SUCCESS;
}


/* Computes the sum of digits of a number.
 *
 * input parameters:	number	number for which the checksum is
 *				requested
 * output parameters:	none
 * return value:	sum of digits of "number"
 * side effects:	none
 *
 */
int checksum (unsigned long long number)
{
  unsigned long long tmp;
  int		     chk_sum;

  tmp	  = number;
  chk_sum = 0;
  while (tmp > 0)
  {
    chk_sum += (tmp % 10);
    tmp	     = tmp / 10;
  }
  return chk_sum;
}
