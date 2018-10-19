/* This program determines all numbers in a defined intervall (from
 * 1 up to at most INT_MAX) which satisfy the following condition:
 * The square of the sum of the digits of the number is equal to the
 * sum of digits of the square of the number. Furthermore the program
 * measures the time needed to compute the numbers. This version
 * "distributes" the work dynamically among all processes, i.e., every
 * process gets as much work as it can deal with while other processes
 * are working.
 *
 * The master process (rank 0) waits for requests (CMD_NEED_WORK) from
 * slave processes. It answers with CMD_SEND_WORK or CMD_EXIT if
 * everything is finished. Each slave process still prints its results
 * on the display itself.
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: checksum_dyn.c			Author: S. Gross
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
#define DEF_CHUNK_SIZE	1000		/* default chunk size		*/
#define	NUM_BLK		1		/* # of blocks in INTERVAL	*/
#define CMD_NEED_WORK	1		/* define some commands		*/
#define CMD_SEND_WORK	2
#define CMD_EXIT	3

typedef struct interval
{
  long from,				/* first number			*/
       to;				/* last number			*/
} INTERVAL;

static MPI_Datatype structInterval;	/* datatype to pass INTERVAL	*/

int  checksum (unsigned long long number);
void usage (char *prog_name);


int main (int argc, char *argv[])
{
  int		     mytid,		/* my task id			*/
		     ntasks,		/* number of parallel tasks	*/
		     chk_sum,		/* checksum of digits		*/
		     digits_n,		/* # of digits of "upper_bound"	*/
		     digits_sn,		/* # of digits of squared num	*/
		     namelen;		/* length of processor name	*/
  long		     i,			/* loop variable		*/
		     chunk_size,	/* amount of work per request	*/
		     upper_bound;	/* upper bound of intervall	*/
  unsigned long long sq_num;		/* square of a number		*/
  clock_t	     my_CPUtime,	/* used CPU-time		*/
		     sum_CPUtime;
  double	     elapsed_time;	/* elapsed time on root node	*/
  char		     processor_name[MPI_MAX_PROCESSOR_NAME];
  INTERVAL	     new_interval;
  MPI_Status	     status;		/* message details		*/
  /* variables for new datatype INTERVAL: One block with 2 numbers of
   * type "long" starting at offset 0. If the structure contains
   * blocks with different datatypes, you must determine the starting
   * address of each block with "MPI_Get_address ()". Afterwards you
   * must subtract the address of the first block (offsets[0]) from
   * each address to get relative addresses within the structure.
   */
  int 		     blkcnt[NUM_BLK]  = { 2 };
  MPI_Datatype	     types[NUM_BLK]   = { MPI_LONG };
  MPI_Aint	     offsets[NUM_BLK] = { 0 };
  /* If you have a more complicated data structure and the compiler
   * does padding in mysterious ways, the following may be safer (no
   * changes to the above datatype are necessary)
   *
   * int 	     blkcnt[NUM_BLK + 1]  = { 2, 1 };
   * MPI_Datatype    types[NUM_BLK + 1]   = { MPI_LONG, MPI_UB };
   * MPI_Aint	     offsets[NUM_BLK + 1] = { 0, sizeof (INTERVAL) };
   *
   * Now you have set the extent of the datatype explicitly and it
   * doesn't rely on the compiler. This way it should be saver and
   * less error-prone if you replicate the datatype in a send/receive
   * operation. Don't forget the "new block" when you create the new
   * datatype.
   *
   * MPI_Type_create_struct (NUM_BLK + 1, blkcnt, offsets, types,
   *			     &structInterval);
   *
   * You can print the extent of a datatype in the following way:
   *
   * MPI_Aint lb, ext;
   * MPI_Type_get_extent (structInterval, &lb, &ext);
   * printf ("structInterval lower bound: %d    "
   *	     "structInterval extent: %d\n", (int) lb, (int) ext);
   *
   */

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

  /* do an unnecessary initialization to make the GNU compiler happy
   * so that you won't get a warning about the use of a possibly
   * uninitialized variable
   */
  upper_bound = 0;
  chunk_size  = 0;
  if (mytid == 0)
  {
    /* evaluate command line arguments					*/
    switch (argc)
    {
      case 1:
	upper_bound = DEF_UP_BOUND;
	chunk_size  = DEF_CHUNK_SIZE;
	break;

      case 2:
	/* program was called with upper bound of intervall		*/
	upper_bound = atol (argv[1]);
	chunk_size  = DEF_CHUNK_SIZE;
	if ((upper_bound <= 0) || (upper_bound > INT_MAX))
	{
	  fprintf (stderr, "\n\nError: The parameter must be between "
		   "1 and %d\n", INT_MAX);
	  usage (argv[0]);
	}
	break;

      case 3:
	/* program was called with upper bound of intervall and chunk
	 * size
	 */
	upper_bound = atol (argv[1]);
	chunk_size  = atol (argv[2]);
	if ((upper_bound <= 0) || (upper_bound > INT_MAX) ||
	    (chunk_size <= 0) || (chunk_size > INT_MAX))
	{
	  fprintf (stderr, "\n\nError: Both parameters must be between "
		   "1 and %d\n", INT_MAX);
	  usage (argv[0]);
	}
	break;

      default:
	fprintf (stderr, "\n\nError: Too many parameters.\n");
	upper_bound = -1;
	chunk_size  = -1;
	usage (argv[0]);
    }
    /*determine how many digits the largest number can have		*/
    digits_n  = (int) (log10 ((double) (upper_bound) + 1));
    digits_sn = (int) (log10 ((double) (upper_bound * upper_bound) + 1));
  }

  /* Now we can start our real work!					*/
  my_CPUtime   = clock ();
  /* create new datatype to send/receive the bounds of an interval	*/
  MPI_Type_create_struct (NUM_BLK, blkcnt, offsets, types,
			  &structInterval);
  MPI_Type_commit (&structInterval);
  /* Each task  m u s t  call MPI_Bcast to learn the values of
   * "digits_n" and "digits_sn"
   */
  MPI_Bcast (&digits_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&digits_sn, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (mytid == 0)
  {
    /* master process							*/
    if ((upper_bound <= 0) || (upper_bound > INT_MAX) ||
	(chunk_size <= 0) || (chunk_size > INT_MAX))
    {
      /* Error: terminate all other processes				*/
      for (i = 1; i < ntasks; ++i)
      {
	/* MPI_Recv (buffer, count, datatype, source, ...);		*/
	MPI_Recv ((char *) NULL, 0, structInterval, MPI_ANY_SOURCE,
		  MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	fprintf (stderr,"I terminate process with rank %d\n",
		 status.MPI_SOURCE);
	/* MPI_Send (buffer, count, datatype, destination, ...);	*/
	MPI_Send ((char *) NULL, 0, structInterval, status.MPI_SOURCE,
		  CMD_EXIT, MPI_COMM_WORLD);
      }
      MPI_Finalize ();
      exit (EXIT_SUCCESS);
    }
    else
    {
      for (i = 1; i <= upper_bound; i += chunk_size)
      {
	/* MPI_Recv (buffer, count, datatype, source, ...);		*/
	MPI_Recv ((char *) NULL, 0, structInterval, MPI_ANY_SOURCE,
		  MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	if (status.MPI_TAG == CMD_NEED_WORK)
	{
	  new_interval.from = i;
	  new_interval.to   = i + chunk_size - 1;
	  if (new_interval.to > upper_bound)
	  {
	    new_interval.to = upper_bound;
	  }
	  /* MPI_Send (buffer, count, datatype, destination, ...);	*/
	  MPI_Send (&new_interval, 1, structInterval,
		    status.MPI_SOURCE, CMD_SEND_WORK, MPI_COMM_WORLD);
	}
	else
	{
	  fprintf (stderr, "\n\nError: Process with rank %d used "
		   "unknown tag.\n"
		   "I terminate the process.\n\n", status.MPI_SOURCE);
	  /* MPI_Send (buffer, count, datatype, destination, ...);	*/
	  MPI_Send ((char *) NULL, 0, structInterval,
		    status.MPI_SOURCE, CMD_EXIT, MPI_COMM_WORLD);
	}
      }
      /* all work done -> terminate all processes			*/
      for (i = 1; i < ntasks; ++i)
      {
	/* MPI_Recv (buffer, count, datatype, source, ...);		*/
	MPI_Recv ((char *) NULL, 0, structInterval, MPI_ANY_SOURCE,
		  MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	/* MPI_Send (buffer, count, datatype, destination, ...);	*/
	MPI_Send ((char *) NULL, 0, structInterval, status.MPI_SOURCE,
		  CMD_EXIT, MPI_COMM_WORLD);
      }
    }
  }
  else
  {
    /* slave process							*/
    int      more_to_do;

    more_to_do = 1;
    while (more_to_do == 1)
    {
      /* ask for some work
       * MPI_Send (buffer, count, datatype, destination, ...);
       */
      MPI_Send ((char *) NULL, 0, structInterval, 0, CMD_NEED_WORK,
		MPI_COMM_WORLD);
      /* MPI_Recv (buffer, count, datatype, source, ...);		*/
      MPI_Recv (&new_interval, 1, structInterval, 0, MPI_ANY_TAG,
		MPI_COMM_WORLD, &status);
      if (status.MPI_TAG != CMD_EXIT)
      {
	for (i = new_interval.from; i <= new_interval.to; ++i)
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
      }
      else
      {
	more_to_do = 0;			/* terminate			*/
      }
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


/* Prints a message how to call this program.
 *
 * input parameters:	prog_name	name of this program
 * output parameters:	none
 * return value:	none
 * side effects:	none
 *
 */
void usage (char *prog_name)
{
  fprintf (stderr, "Usage:\n"
	   "  mpiexec -np <number of processes> %s \\ \n"
	   "\t[<upper bound>] [<chunk size>]\n",
	   prog_name);
}
