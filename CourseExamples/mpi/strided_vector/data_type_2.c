/* The program demonstrates how to set up and use a strided vector.
 * The process with rank 0 creates a matrix. The columns of the
 * matrix will then be distributed with a collective communication
 * operation to all processes. Each process performs an operation on
 * all column elements. Afterwards the results are collected in the
 * source matrix overwriting the original column elements.
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
 * "MPI_Type_vector" and uses collective communication. The new
 * data type knows the number of elements within one column and the
 * spacing between two column elements. The program needs exactly
 * n processes if the matrix has n columns.
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: data_type_2.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define	P		6		/* # of rows			*/
#define Q		10		/* # of columns			*/
#define FACTOR		2		/* multiplicator for col. elem.	*/

static void print_matrix (int p, int q, double **mat);


int main (int argc, char *argv[])
{
  int    ntasks,			/* number of parallel tasks	*/
         mytid,				/* my task id			*/
         namelen,			/* length of processor name	*/
         i, j,				/* loop variables		*/
         tmp;				/* temporary value		*/
  double matrix[P][Q],
	 column[P];
  char   processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Datatype	column_t,		/* column type (strided vector)	*/
		tmp_column_t;		/* needed to resize the extent	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  /* check that we have the correct number of processes in our universe	*/
  if (ntasks != Q)
  {
    /* wrong number of processes					*/
    if (mytid == 0)
    {
      fprintf (stderr, "\nI need %d processes.\n\n"
	       "Usage:\n"
	       "  mpiexec -np %d %s\n\n", Q, Q, argv[0]);
    }
    MPI_Finalize ();
    exit (EXIT_SUCCESS);
  }
  /* Now let's start with the real work					*/
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

  /* Build the new type for a strided vector and resize the extent of
   * the new datatype in such a way that the extent of the whole column
   * looks like just one element so that the next column starts in
   * matrix[0][i] in MPI_Scatter/MPI_Gather.
   */
  MPI_Type_vector (P, 1, Q, MPI_DOUBLE, &tmp_column_t);
  MPI_Type_create_resized (tmp_column_t, 0, sizeof (double), &column_t);
  MPI_Type_commit (&column_t);
  MPI_Type_free (&tmp_column_t);
  if (mytid == 0)
  {
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
  }
  /* distribute columns							*/
  MPI_Scatter (matrix, 1, column_t, column, P, MPI_DOUBLE, 0,
	       MPI_COMM_WORLD);
  /* modify column elements						*/
  for (i = 0; i < P; ++i)
  {
    if ((mytid % 2) == 0)
    {
      column[i] *= column[i];
    }
    else
    {
      column[i] *= FACTOR;
    }
  }
  /* collect result vectors		     				*/
  MPI_Gather (column, P, MPI_DOUBLE, matrix, 1, column_t, 0,
	      MPI_COMM_WORLD);
  if (mytid == 0)
  {
    printf ("\n\nresult matrix:\n"
	    "(odd columns: elements squared; even columns: elements "
	    "multiplied with %d)\n\n", FACTOR);
    print_matrix (P, Q, (double **) matrix);
  }
  MPI_Type_free (&column_t);
  MPI_Finalize ();
  return EXIT_SUCCESS;
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
