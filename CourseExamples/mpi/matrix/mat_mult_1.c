/* Matrix multiplication: c = a * b.
 *
 * This program needs as many processes as there are rows in
 * matrix "a".
 *
 *
 * Compiling:
 *   mpicc -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: mat_mult_1.c		       	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define	P		4		/* # of rows			*/
#define Q		6		/* # of columns / rows		*/
#define R		8		/* # of columns			*/

static void print_matrix (int p, int q, double **mat);

int main (int argc, char *argv[])
{
  int	 ntasks,			/* number of parallel tasks	*/
	 mytid,				/* my task id			*/
	 namelen,			/* length of processor name	*/
	 i, j, k,			/* loop variables		*/
	 tmp;				/* temporary value		*/
  double a[P][Q], b[Q][R],		/* matrices to multiply		*/
	 c[P][R],			/* matrix for result		*/
	 row_a[Q],			/* one row of matrix "a"	*/
	 row_c[R];			/* one row of matrix "c"	*/
  char	 processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
  /* check that we have the correct number of processes in our universe	*/
  if (ntasks != P)
  {
    /* wrong number of processes					*/
    if (mytid == 0)
    {
      fprintf (stderr, "\nI need %d processes.\n\n"
	       "Usage:\n"
	       "  mpiexec -np %d %s\n\n", P, P, argv[0]);
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
  if (mytid == 0)
  {
    tmp = 1;
    for (i = 0; i < P; ++i)		/* initialize matrix a		*/
    {
      for (j = 0; j < Q; ++j)
      {
	a[i][j] = tmp++;
      }
    }
    printf ("\n\n(%d,%d)-matrix a:\n\n", P, Q);
    print_matrix (P, Q, (double **) a);
    tmp = Q * R;
    for (i = 0; i < Q; ++i)		/* initialize matrix b		*/
    {
      for (j = 0; j < R; ++j)
      {
	b[i][j] = tmp--;
      }
    }
    printf ("(%d,%d)-matrix b:\n\n", Q, R);
    print_matrix (Q, R, (double **) b);
  }
  /* send matrix "b" to all processes					*/
  MPI_Bcast (b, Q * R, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  /* send row i of "a" to process i					*/
  MPI_Scatter (a, Q, MPI_DOUBLE, row_a, Q, MPI_DOUBLE, 0,
	       MPI_COMM_WORLD);
  for (j = 0; j < R; ++j)		/* compute i-th row of "c"	*/
  {
    row_c[j] = 0.0;
    for (k = 0; k < Q; ++k)
    {
      row_c[j] = row_c[j] + row_a[k] * b[k][j];
    }
  }
  /* receive row i of "c" from process i				*/
  MPI_Gather (row_c, R, MPI_DOUBLE, c, R, MPI_DOUBLE, 0,
	      MPI_COMM_WORLD);
  if (mytid == 0)
  {
    printf ("(%d,%d)-result-matrix c = a * b :\n\n", P, R);
    print_matrix (P, R, (double **) c);
  }
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

