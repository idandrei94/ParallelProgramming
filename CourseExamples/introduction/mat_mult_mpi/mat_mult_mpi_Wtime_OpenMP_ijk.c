/* Matrix multiplication: c = a * b.
 *
 * The program uses between one and n processes to multiply the
 * matrices if matrix "a" has n rows. If you start the program with
 * one process it has to compute all n rows of the result matrix alone
 * and if you start the program with n processes each process computes
 * one row of the result matrix. Every process must know how many rows
 * it has to work on so that it can allocate enough buffer space for
 * its rows. Therefore the process with rank 0 computes the numbers of
 * rows for each process in the array "num_rows" and distributes this
 * array with MPI_Broadcast to all processes. Each process can now
 * allocate memory for its row block in "row_a" and "row_c". There
 * is still one task to do before the rows of matrix "a" can be
 * distributed with MPI_Scatterv: The size of each row block and the
 * offset of each row block must be computed und must be stored in
 * the arrays "sr_counts" and "sr_disps". These data structures must
 * be adjusted to the corresponding values of result matrix "c" before
 * you can call MPI_Gatherv to collect the row blocks of the result
 * matrix.
 *
 * You must set the environment variable OMP_NUM_THREADS prior to the
 * execution of the program, e.g., "setenv OMP_NUM_THREADS 8" to create
 * eight parallel threads.
 *
 *
 * Compiling:
 *   mpicc {-fopenmp | -qopenmp | -xopenmp}
 *     [-DP=<value>] [-DQ=<value>] [-DR=<value>] \
 *     -o <program name> <source code file name> -lm
 *
 * Running:
 *   mpiexec -np <number of processes> <program name>
 *
 *
 * File: mat_mult_mpi_Wtime_OpenMP_ijk.c	Author: S. Gross
 * Date: 04.08.2017
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "mpi.h"

#define EPS	DBL_EPSILON		/* from float.h (2.2...e-16)	*/

#ifndef P
  #define P	1984			/* # of rows			*/
#endif
#ifndef Q
  #define Q	1984			/* # of columns / rows		*/
#endif
#ifndef R
  #define R	1984			/* # of columns			*/
#endif

/* define macro to test the result of a "malloc" operation		*/
#define TestEqualsNULL(val)  \
  if (val == NULL) \
  { \
    fprintf (stderr, "file: %s  line %d: Couldn't allocate memory.\n", \
	     __FILE__, __LINE__); \
    exit (EXIT_FAILURE); \
  }


/* matrices of this size are too large for a normal stack size and
 * must be allocated globally or with the keyword "static".
 */
static double a[P][Q], b[Q][R],		/* matrices to multiply		*/
	      c[P][R];			/* result matrix		*/


int main (int argc, char *argv[])
{
  int	  ntasks,			/* number of parallel tasks	*/
	  mytid,			/* my task id			*/
	  i, j, k,			/* loop variables		*/
	  *num_rows,			/* # of rows in a row block	*/
	  *sr_counts = NULL,		/* send/receive counts		*/
	  *sr_disps = NULL,		/* send/receive displacements	*/
	  ok;				/* temporary value		*/
  double  tmp,				/* temporary value		*/
	  **row_a,			/* row block of matrix "a"	*/
	  **row_c,			/* row block of matrix "c"	*/
	  t_comm, t_comm_tmp,		/* time for communication	*/
	  t_mult_comp, t_mult_tmp;	/* time for multiplication	*/
  time_t  st_mult_ab, et_mult_ab;	/* start/end time (wall clock)	*/

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &mytid);
  MPI_Comm_size (MPI_COMM_WORLD, &ntasks);

  if ((ntasks > P) && (mytid == 0))
  {
    fprintf (stderr, "\n\nI can use at most %d processes.\n"
	     "Usage:\n"
	     "  LAM-MPI: mpiexec -np n %s\n"
	     "  OpenMPI: mpiexec -mca btl ^udapl -np n %s\n"
	     "  MPICH2:  mpiexec -np n %s\n"
	     "where \"n\" is a value between 1 and %d.\n\n",
	     P, argv[0], argv[0], argv[0], P);
  }
  if (ntasks > P)
  {
    MPI_Finalize ();
    exit (EXIT_SUCCESS);
  }


  if (mytid == 0)
  {
    #pragma omp parallel default(none) private(i, j) shared(a, b)
    {
      /* initialize matrix "a"						*/
      #pragma omp for
      for (i = 0; i < P; ++i)
      {
	for (j = 0; j < Q; ++j)
        {
	  a[i][j] = 2.0;
	}
      }

      /* initialize matrix "b"						*/
      #pragma omp for
      for (i = 0; i < Q; ++i)
      {
        for (j = 0; j < R; ++j)
        {
	  b[i][j] = 3.0;
	}
      }
    }
  }

  /* Compute result matrix "c" and measure some times.
   * At first matrix "b" and a block of matrix "a" must be
   * distributed to all processes. This will be measured
   * as "communication time".
   */
  st_mult_ab = time (NULL);
  t_comm_tmp = MPI_Wtime ();
  /* send matrix "b" to all processes					*/
  MPI_Bcast (b, Q * R, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  /* allocate memory for array containing the number of rows of a
   * row block for each process
   */
  num_rows = (int *) malloc ((size_t) ntasks * sizeof (int));
  TestEqualsNULL (num_rows);		/* test if memory was available	*/
  if (mytid == 0)
  {
    /* allocate memory for arrays containing the size and
     * displacement of each row block
     */
    sr_counts = (int *) malloc ((size_t) ntasks * sizeof (int));
    TestEqualsNULL (sr_counts);
    sr_disps = (int *) malloc ((size_t) ntasks * sizeof (int));
    TestEqualsNULL (sr_disps);
    /* compute number of rows in row block for each process		*/
    tmp = P / ntasks;
    for (i = 0; i < ntasks; ++i)
    {
      num_rows[i] = (int) tmp;		/* number of rows		*/
    }
    for (i = 0; i < (P % ntasks); ++i)	/* adjust size		 	*/
    {
      num_rows[i]++;
    }
    for (i = 0; i < ntasks; ++i)
    {
      sr_counts[i] = num_rows[i] * Q;	/* size of row block		*/
    }
    sr_disps[0] = 0;			/* start of i-th row block	*/
    for (i = 1; i < ntasks; ++i)
    {
      sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
    }
  }
  /* inform all processes about their amount of data to work on		*/
  MPI_Bcast (num_rows, ntasks, MPI_INT, 0, MPI_COMM_WORLD);
  row_a =
   (double **) malloc ((size_t) (num_rows[mytid] * Q) * sizeof (double)); 
  TestEqualsNULL (row_a);
  row_c =
   (double **) malloc ((size_t) (num_rows[mytid] * R) * sizeof (double)); 
  TestEqualsNULL (row_c);
  /* send row block i of "a" to process i				*/
  MPI_Scatterv (a, sr_counts, sr_disps, MPI_DOUBLE, row_a,
		num_rows[mytid] * Q, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  t_comm = MPI_Wtime () - t_comm_tmp;

  /* Compute row block i of result matrix "c". The compiler doesn't
   * know the structure of the row blocks row_a and row_c so that you
   * have to do the index calculations for row_c[i][j] and row_a[i][k]
   * yourself. In C a matrix mat is stored row-by-row so that the i-th
   * row starts at location "i * q" if the matrix has "q" columns.
   * Therefore the address of mat[i][j] can be expressed as
   * "(double *) mat + i * q + j" and mat[i][j] itself as
   * "*((double *) mat + i * q + j)".
   */
  t_mult_tmp = MPI_Wtime ();
  #pragma omp parallel for default(none) private(i, j, k, mytid)	\
    shared(num_rows, row_a, row_c, b)
  for (i = 0; i < num_rows[mytid]; ++i)
  {
    for (j = 0; j < R; ++j)
    {
      *((double *) row_c + i * R + j) = 0.0;	/* row_c[i][j] = 0.0	*/
      for (k = 0; k < Q; ++k)
      {
	/* row_c[i][j] += row_a[i][k] * b[k][j]				*/

	*((double *) row_c + i * R + j) +=
	  *((double *) row_a + i * Q + k) * b[k][j];
      }
    }
  }
  t_mult_tmp = MPI_Wtime () - t_mult_tmp;
  /* Now the result matrix "c" must be collected. This will
   * once more be measured as "communication time".
   */
  t_comm_tmp = MPI_Wtime ();
  if (mytid == 0)
  {
    /* adjust "sr_counts" and "sr_disps" for matrix "c"			*/
    for (i = 0; i < ntasks; ++i)
    {
      sr_counts[i] = num_rows[i] * R;	/* size of row block		*/
    }
    for (i = 1; i < ntasks; ++i)	/* start of i-th row block	*/
    {
      sr_disps[i] = sr_disps[i - 1] + sr_counts[i - 1];
    }
  }
  /* receive row block i of "c" from process i				*/
  MPI_Gatherv (row_c, num_rows[mytid] * R, MPI_DOUBLE, c, sr_counts,
	       sr_disps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  t_comm += (MPI_Wtime () - t_comm_tmp);
  et_mult_ab = time (NULL);
  /* collect the computation time for the matrix multiplication
   * from all processes
   */
  MPI_Reduce (&t_mult_tmp, &t_mult_comp, 1, MPI_DOUBLE, MPI_SUM,
	      0, MPI_COMM_WORLD);

  if (mytid == 0)
  {
    /* test values of matrix "c"					*/
    tmp = c[0][0];
    ok  = 0;
    #pragma omp parallel for default(none) private(i, j) \
      shared(c, tmp) reduction(+:ok)
    for (i = 0; i < P; ++i)
    {
      for (j = 0; j < R; ++j)
      {
	if (fabs (c[i][j] - tmp) > EPS)
        {
	  ok++;
	}
      }
    }

    if (ok == 0)
    {
      printf ("c[%d][%d] = a[%d][%d] * b[%d][%d] was successful.\n",
	      P, R, P, Q, Q, R);
    }
    else
    {
      printf ("c[%d][%d] = a[%d][%d] * b[%d][%d] was not successful.\n"
	      "%d values differ.\n", P, R, P, Q, Q, R, ok);
    }
    printf ("                      elapsed time      cpu time\n"
	    "Mult \"a\" and \"b\":        %6.2f s\n"
            "  communication time:    %6.2f s\n"
            "  computation time:                    %6.2f s\n",
            difftime (et_mult_ab, st_mult_ab),
            t_comm, t_mult_comp);
    free (sr_counts);
    free (sr_disps);
  }
  free (num_rows);
  free (row_a);
  free (row_c);
  MPI_Finalize ();
  return EXIT_SUCCESS;
}
