/* The program demonstrates how to set up and use a strided vector.
 * The process with rank 0 creates a matrix. The columns of the
 * matrix will then be distributed with a collective communication
 * operation to all processes. Each process performs an operation on
 * all column elements. Afterwards the results are collected in the
 * source matrix overwriting the original column elements.
 *
 * The program uses between one and n processes to change the values
 * of the column elements if the matrix has n columns. If you start
 * the program with one process it has to work on all n columns alone
 * and if you start it with n processes each process modifies the
 * values of one column. Every process must know how many columns it
 * has to modify so that it can allocate enough buffer space for its
 * column block. Therefore the process with rank 0 computes the
 * numbers of columns for each process in the array "num_columns" and
 * distributes this array with MPI_Broadcast to all processes. Each
 * process can now allocate memory for its column block. There is
 * still one task to do before the columns of the matrix can be
 * distributed with MPI_Scatterv: The size of every column block and
 * the offset of every column block must be computed und stored in
 * the arrays "sr_counts" and "sr_disps".
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
 * spacing between two column elements. The program uses at most
 * n processes if the matrix has n columns, i.e. depending on the
 * number of processes each process receives between 1 and n columns.
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: data_type_3.c			Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define	P		6		/* # of rows			*/
#define Q		10		/* # of columns			*/
#define FACTOR		2		/* multiplicator for col. elem.	*/

/* define macro to test the result of a "malloc" operation		*/
#define TestEqualsNULL(val)  \
  if (val == NULL) \
  { \
    fprintf (stderr, "file: %s  line %d: Couldn't allocate memory.\n", \
	     __FILE__, __LINE__); \
    exit (EXIT_FAILURE); \
  }


static void print_matrix (int p, int q, double **mat);


int main (int argc, char *argv[])
{
  int    ntasks,			/* number of parallel tasks	*/
         mytid,				/* my task id			*/
         namelen,			/* length of processor name	*/
         i, j,				/* loop variables		*/
	 *num_columns,			/* # of columns in column block	*/
	 *sr_counts,			/* send/receive counts		*/
	 *sr_disps,			/* send/receive displacements	*/
	 tmp, tmp1;			/* temporary values		*/
  double matrix[P][Q],
    	 **column_block;		/* column block of matrix	*/
  char   processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Datatype	column_t,		/* column type (strided vector)	*/
		column_block_t,
		tmp_column_t;		/* needed to resize the extent	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  /* check that we have the correct number of processes in our universe	*/
  if (ntasks > Q)
  {
    /* wrong number of processes					*/
    if (mytid == 0)
    {
      fprintf (stderr, "\nI can use at most %d processes.\n"
	       "Usage:\n"
	       "  mpiexec -np n %s\n"
	       "where \"n\" is a value between 1 and %d.\n\n",
	       Q, argv[0], Q);
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
   * matrix[0][i] in MPI_Scatterv/MPI_Gatherv.
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
  /* allocate memory for array containing the number of columns of a
   * column block for each process
   */
  num_columns = (int *) malloc ((size_t) ntasks * sizeof (int));
  TestEqualsNULL (num_columns);		/* test if memory was available	*/

  /* do an unnecessary initialization to make the GNU compiler happy
   * so that you won't get a warning about the use of a possibly
   * uninitialized variable
   */
  sr_counts = NULL;
  sr_disps  = NULL;
  if (mytid == 0)
  {
    /* allocate memory for arrays containing the size and
     * displacement of each column block
     */
    sr_counts = (int *) malloc ((size_t) ntasks * sizeof (int));
    TestEqualsNULL (sr_counts);		/* test if memory was available	*/
    sr_disps = (int *) malloc ((size_t) ntasks * sizeof (int));
    TestEqualsNULL (sr_disps);		/* test if memory was available	*/
    /* compute number of columns in column block for each process	*/
    tmp = Q / ntasks;
    for (i = 0; i < ntasks; ++i)
    {
      num_columns[i] = tmp;		/* number of columns		*/
    }
    for (i = 0; i < (Q % ntasks); ++i)	/* adjust size		 	*/
    {
      num_columns[i]++;
    }
    for (i = 0; i < ntasks; ++i)
    {
      /* nothing to do because "column_t" contains already all
       * elements of a column, i.e., the size is equal to the
       * number of columns in the block
       */
      sr_counts[i] = num_columns[i];	/* size of column-block		*/
    }
    sr_disps[0] = 0;			/* start of i-th column-block	*/
    for (i = 1; i < ntasks; ++i)
    {
      sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
    }
  }
  /* inform all processes about their column block sizes		*/
  MPI_Bcast (num_columns, ntasks, MPI_INT, 0, MPI_COMM_WORLD);
  /* allocate memory for a column block and define a new derived data
   * type for the column block. This data type is possibly different
   * for different processes if the number of processes isn't a factor
   * of the row size of the original matrix. Don't forget to resize the
   * extent of the new data type in such a way that the extent of the
   * whole column looks like just one element so that the next column
   * starts in column_block[0][i] in MPI_Scatterv/MPI_Gatherv.
   */
  column_block =
    (double **) malloc ((size_t) (P * num_columns[mytid]) *
				  sizeof (double));
  TestEqualsNULL (column_block);	/* test if memory was available	*/
  MPI_Type_vector (P, 1, num_columns[mytid], MPI_DOUBLE, &tmp_column_t);
  MPI_Type_create_resized (tmp_column_t, 0, sizeof (double),
			   &column_block_t);
  MPI_Type_commit (&column_block_t);
  MPI_Type_free (&tmp_column_t);
  /* send column block i of "matrix" to process i			*/
  MPI_Scatterv (matrix, sr_counts, sr_disps, column_t,
		column_block, num_columns[mytid], column_block_t,
		0, MPI_COMM_WORLD);
  /* Modify column elements. The compiler doesn't know the structure
   * of the column block matrix so that you have to do the index
   * calculations for mat[i][j] yourself. In C a matrix is stored
   * row-by-row so that the i-th row starts at location "i * q" if
   * the matrix has "q" columns. Therefore the address of mat[i][j]
   * can be expressed as "(double *) mat + i * q + j" and mat[i][j]
   * itself as "*((double *) mat + i * q + j)".
   */
  for (i = 0; i < P; ++i)
  {
    for (j = 0; j < num_columns[mytid]; ++j)
    {
      if ((mytid % 2) == 0)
      {
	/* column_block[i][j] *= column_block[i][j]			*/

	*((double *) column_block + i * num_columns[mytid] + j) *=
	  *((double *) column_block + i * num_columns[mytid] + j);
      }
      else
      {
	/* column_block[i][j] *= FACTOR					*/

	*((double *) column_block + i * num_columns[mytid] + j) *=
	  FACTOR;
      }
    }
  }
  /* receive column-block i of "matrix" from process i			*/
  MPI_Gatherv (column_block, num_columns[mytid], column_block_t,
	       matrix, sr_counts, sr_disps, column_t,
	       0, MPI_COMM_WORLD);
  if (mytid == 0)
  {
    printf ("\n\nresult matrix:\n"
	    "  elements are sqared in columns:\n  ");
    tmp  = 0;
    tmp1 = 0;
    for (i = 0; i < ntasks; ++i)
    {
      tmp1 = tmp1 + num_columns[i];
      if ((i % 2) == 0)
      {
	for (j = tmp; j < tmp1; ++j)
        {
	  printf ("%4d", j);
	}
      }
      tmp = tmp1;
    }
    printf ("\n  elements are multiplied with %d in columns:\n  ",
	    FACTOR);
    tmp  = 0;
    tmp1 = 0;
    for (i = 0; i < ntasks; ++i)
    {
      tmp1 = tmp1 + num_columns[i];
      if ((i % 2) != 0)
      {
	for (j = tmp; j < tmp1; ++j)
        {
	  printf ("%4d", j);
	}
      }
      tmp = tmp1;
    }
    printf ("\n\n\n");
    print_matrix (P, Q, (double **) matrix);
    free (sr_counts);
    free (sr_disps);
  }
  free (num_columns);
  free (column_block);
  MPI_Type_free (&column_t);
  MPI_Type_free (&column_block_t);
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
