/* Small program  that creates and prints column vectors of a matrix.
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
 *
 * Compiling:
 *   mpicc -o column_double column_double.c
 *
 * Running:
 *   mpiexec -np <number of processes> column_double
 *
 *
 * File: column_double.c		Author: S. Gross
 * Date: 19.07.2018
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define	P		  4		/* # of rows			*/
#define Q		  6		/* # of columns			*/
#define NUM_ELEM_PER_LINE 6		/* to print a vector		*/


int main (int argc, char *argv[])
{
  int    ntasks,			/* number of parallel tasks	*/
         mytid,				/* my task id			*/
         i, j,				/* loop variables		*/
	 tmp;				/* temporary value		*/
  double matrix[P][Q],
	 column[P];
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
  /* Build the new type for a strided vector and resize the extent
   * of the new datatype in such a way that the extent of the whole
   * column looks like just one element.
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
    printf ("\nmatrix:\n\n");		/* print matrix			*/
    for (i = 0; i < P; ++i)
    {
      for (j = 0; j < Q; ++j)
      {
	printf ("%6g", matrix[i][j]);
      }
      printf ("\n");
    }
    printf ("\n");
  }
  MPI_Scatter (matrix, 1, column_t, column, P, MPI_DOUBLE, 0,
	       MPI_COMM_WORLD);
  /* Each process prints its column. The output will intermingle on
   * the screen so that you must use "-output-filename" in Open MPI.
   */
  printf ("\nColumn of process %d:\n", mytid);
  for (i = 0; i < P; ++i)
  {
    if (((i + 1) % NUM_ELEM_PER_LINE) == 0)
    {
      printf ("%6g\n", column[i]);
    }
    else
    {
      printf ("%6g", column[i]);
    }
  }
  printf ("\n");
  MPI_Type_free (&column_t);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
