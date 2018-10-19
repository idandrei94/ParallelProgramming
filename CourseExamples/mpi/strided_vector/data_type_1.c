/* The program demonstrates how to set up and use a strided vector.
 * The process with rank 0 (master process) creates a matrix. The
 * columns of the matrix will then be distributed with point-to-point
 * communication operations to all other processes (slave processes)
 * which perform an operation on all elements of the vector and send
 * the result back. The result vectors are collected in the source
 * matrix overwriting the original column elements.
 *
 * An MPI data type is defined by its size, its contents, and its
 * extent. When multiple elements of the same size are used in a
 * contiguous manner (e.g. in a "scatter" operation or an operation
 * with "count" greater than one) the extent is used to compute where
 * the next element will start. The extent for a derived data type is
 * as big as the size of the derived data type so that the first
 * elements of the second structure will start after the last element
 * of the first structure, i.e., you have to "resize" the new data
 * type if you want to send it multiple times (count > 1) or to
 * scatter/gather it to many processes. Restrict the extent of the
 * derived data type for a strided vector in such a way that it looks
 * like just one element if it is used with "count > 1" or in a
 * scatter/gather operation.
 *
 * This version constructs a new column type (strided vector) with
 * "MPI_Type_vector" and uses point-to-point communication. The new
 * data type knows the number of elements within one column and the
 * spacing between two column elements. That means that you must
 * send and receive just one element of the new data type to send or
 * receive a whole column. The program needs at least two processes.
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: data_type_1.c		       	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define	P		6		/* # of rows			*/
#define Q		10		/* # of columns			*/
#define FACTOR		2		/* multiplicator for col. elem.	*/
#define	CMD_SQ		1		/* command: square each element	*/
#define CMD_MPY		2		/* multiply elem. with FACTOR	*/
#define	CMD_EXIT	30		/* terminate			*/
#define	CMD_RESULT	40		/* sending the result vector	*/

static MPI_Datatype	column_t;	/* column type (strided vector)	*/

static void master (char *prog_name);
static void slave (void);
static void print_matrix (int p, int q, double **mat);


int main (int argc, char *argv[])
{
  int  ntasks,				/* number of parallel tasks	*/
       mytid,				/* my task id			*/
       namelen;				/* length of processor name	*/
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init (&argc, &argv);
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

  /* build the new type for a strided vector				*/
  MPI_Type_vector (P, 1, Q, MPI_DOUBLE, &column_t);
  MPI_Type_commit (&column_t);
  if (mytid == 0)
  {
    master (argv[0]);
  }
  else
  {
    slave ();
  }
  MPI_Type_free (&column_t);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}


/* Function for the "master task". The master sends each slave a
 * column of a matrix and requests an operation on the vector
 * elements. If there are more columns than slaves, it sends some
 * or all slaves additional columns after receiving the result
 * for the last request. The master sends all slaves a termination
 * command when all work is done.
 *
 * input parameters:	prog_name	name of this program
 * output parameters:	none
 * return value:	none
 * side effects:	none
 *
 */
void master (char *prog_name)
{
  int		ntasks,			/* number of parallel tasks	*/
		mytid,			/* my task id			*/
		i, j,			/* loop variables		*/
		tmp,			/* temporary value		*/
		runs,			/* # of required runs		*/
		nworker, orig_worker;	/* # of working slave tasks	*/
  double	matrix[P][Q];
  MPI_Status	stat;			/* message details		*/

  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  if (ntasks < 2)
  {
    fprintf (stderr, "\nI need at least two processes.\n"
	     "Usage:\n"
	     "  mpiexec -np n %s\n"
	     "where \"n\" is a value greater than or equal to 2.\n\n",
	     prog_name);
    return;
  }
  if (ntasks > Q + 1)
  {
    nworker = Q;			/* one slave for each column	*/
    printf ("\nYou have started %d processes and this program needs\n"
	    "at most %d processes. I use only processes with ranks\n"
	    "0 to %d. All additional processes are unused.\n",
	    ntasks, Q + 1, Q);
  }
  else
  {
    nworker = ntasks - 1;		/* 1 master task		*/
  }
  /* save "original workers" because "nworker" must be updated
   * in the last run if "Q" isn't a multiple of "nworker"
   */
  orig_worker = nworker;
  runs = Q / nworker;			/* # of necessary runs		*/
  if (runs * nworker != Q)
  {
    runs++;
  }
  tmp = 1;
  for (i = 0; i < P; ++i)		/* initialize matrix		*/
  {
    for (j = 0; j < Q; ++j)
    {
      matrix[i][j] = tmp++;
    }
  }
  printf ("\n\noriginal matrix:\n\n");
  print_matrix (P, Q, (double **) matrix);
  printf ("number of workers: %d\n"
	  "number of runs: %d\n", nworker, runs);
  /* send columns to slave tasks					*/
  for (i = 0; i < runs; ++i)
  {
    if (i == (runs - 1))		/* last run			*/
    {
      if ((Q % nworker) != 0)		/* all workers needed ?		*/
      {
        nworker = Q % nworker;		/* no				*/
      }
    }
    for (j = 1; j <= nworker; ++j)
    {
      /* Each slave from "orig_worker" slaves has processed one
       * column in every "run", so that "i * orig_worker" columns
       * are already processed. "-1" is necessary because the slaves
       * start with rank "1" and the columns with index "0".
       */
      tmp = i * orig_worker + j - 1;
      if ((tmp % 2) == 0)
      {
	MPI_Send (&matrix[0][tmp], 1, column_t, j, CMD_SQ,
		  MPI_COMM_WORLD);
      }
      else
      {
	MPI_Send (&matrix[0][tmp], 1, column_t, j, CMD_MPY,
		  MPI_COMM_WORLD);
      }
    }
    /* wait for result vectors		     				*/
    for (j = 1; j <= nworker; ++j)
    {
      /* worker j has computed column "j - 1" in the i-th run		*/
      tmp = i * orig_worker + j - 1;
      MPI_Recv (&matrix[0][tmp], 1, column_t, j, MPI_ANY_TAG,
		MPI_COMM_WORLD, &stat);
    }
  }
  printf ("\n\nresult matrix:\n"
	  "(odd columns: elements squared; even columns: elements "
	  "multiplied with %d)\n\n", FACTOR);
  print_matrix (P, Q, (double **) matrix);
  /* terminate all slave tasks (worker processes and unused processes)	*/
  for (i = 1; i < ntasks; ++i)
  {
    MPI_Send ((char *) NULL, 0, MPI_CHAR, i, CMD_EXIT, MPI_COMM_WORLD);
  }
}


/* Function for "slave tasks". A slave receives a vector from the
 * master and performs an operation on all elements. It sends the
 * result back to the master.
 *
 * input parameters:	none
 * output parameters:	none
 * return value:	none
 * side effects:	none
 *
 */
void slave (void)
{
  int		i,			/* loop variable		*/
		more_to_do;
  double	column[P];
  MPI_Status	stat;			/* message details		*/

  more_to_do = 1;
  while (more_to_do == 1)
  {
    /* wait for a message from the master task				*/
    MPI_Recv (column, P, MPI_DOUBLE, 0, MPI_ANY_TAG,
	      MPI_COMM_WORLD, &stat);
    if (stat.MPI_TAG != CMD_EXIT)
    {
      for (i = 0; i < P; ++i)
      {
	switch (stat.MPI_TAG)
	{
	  case CMD_SQ:
	    column[i] *= column[i];
	    break;

	  case CMD_MPY:
	    column[i] *= FACTOR;
	    break;

	  default:
	    ;				/* unknown tag -> do nothing	*/
	}
      }
      /* send result back to master task				*/
      MPI_Send (column, P, MPI_DOUBLE, stat.MPI_SOURCE,
		CMD_RESULT, MPI_COMM_WORLD);
    }
    else
    {
      more_to_do = 0;			/* terminate			*/
    }
  }
}


/* Print the values of an arbitrary 2D-matrix of "double" values. The
 * compiler doesn't know the structure of the matrix so that you have
 * to do the index calculations for mat[i][j] yourself. In C a matrix
 * is stored row-by-row so that the i-th row starts at location "i * q"
 * if the matrix has "q" columns. Therefore the address of mat[i][j]
 * can be expressed as "(double *) mat + i * q + j" and mat[i][j]
 * itself as "*((double *) mat + i * q + j)".
 *
 * input parameters:	p	number of rows
 *			q	number of columns
 *			mat	2D-matrix of "double" values
 * output parameters:	none
 * return value:	none
 * side effects:	none
 *
 */
void print_matrix (int p, int q, double **mat)
{
  int i, j;				/* loop variables		*/

  for (i = 0; i < p; ++i)
  {
    for (j = 0; j < q; ++j)
    {
      printf ("%6g", *((double *) mat + i * q + j));
    }
    printf ("\n");
  }
  printf ("\n");
}
